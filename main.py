import os
import sys
import json
import uuid
from typing import List, Literal, TypedDict, Optional, Dict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_openai import ChatOpenAI

# ---------------------------
# State (so simpel wie möglich)
# ---------------------------

class SimState(TypedDict, total=False):
    cause: str                               # Anlass/Ursache
    company_response: Optional[str]          # letzte Unternehmensantwort
    community_reactions: List[str]           # Log der Kommentar-Wellen (als Textblöcke)
    reputation_score: int                    # 0..100 (Start 50)
    iteration: int                           # Runden-Zähler
    status: Literal["ongoing", "resolved", "catastrophe"]
    last_eval: Dict                          # zuletzt berechnete Bewertung


# ---------------------------
# LLM (klein & günstig, aber flexibel)
# ---------------------------

# Du kannst das Modell anpassen. "gpt-4o-mini" hat gutes Kosten/Nutzen.
MODEL_NAME = os.environ.get("SHITSTORM_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=MODEL_NAME, temperature=0.7)


# ---------------------------
# Hilfsfunktionen
# ---------------------------

def _clamp(n: int, lo: int = 0, hi: int = 100) -> int:
    return max(lo, min(hi, n))


def _anger_from_rep(rep: int) -> int:
    # sehr grobe Heuristik: niedrige Reputation => hohe Wut
    return _clamp(100 - rep, 0, 100)


# ---------------------------------------
# Node: Community Reaction (mit interrupt)
# ---------------------------------------

def community_reaction_node(state: SimState) -> SimState:
    cause = state["cause"]
    rep = state.get("reputation_score", 50)
    anger = _anger_from_rep(rep)
    last_company = state.get("company_response")
    prev = "\n\n---\n".join(state.get("community_reactions", [])[-2:])  # nur wenig Kontext

    # Prompt: erzeugt kurze Social-Media-Kommentare ohne Beleidigungen/Slurs
    sys_prompt = (
        "Du simulierst die breite Social-Media-Community in einer PR-Krise. "
        "Erzeuge realistische, kritische, aber nicht diskriminierende Kommentare. "
        "Vermeide Beleidigungen, Drohungen, Slurs. Keine Ratschläge an das Unternehmen."
    )
    user_prompt = f"""Anlass/Cause: {cause}

Letzte Unternehmensantwort (falls vorhanden):
{last_company if last_company else "(keine bisher)"}

Auszüge vorheriger Reaktionen:
{prev if prev else "(noch keine)"}

Aktueller Ärgerlevel (0-100): {anger}

Aufgabe:
- Erzeuge 5 sehr kurze, unterschiedliche Social-Media-Kommentare (1-2 Sätze je).
- Tonalität: bei hohem Ärgerlevel deutlich kritischer/empörter; bei niedrigem Ärgerlevel gemischt/abkühlend.
- Formatiere als nummerierte Liste 1.-5.
- Keine Ratschläge, keine beleidigende Sprache.
"""

    comments = llm.invoke(
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
    ).content.strip()

    # Log in den State schreiben
    new_log = list(state.get("community_reactions", []))
    new_log.append(comments)

    # Jetzt Human-in-the-Loop: nach Unternehmensantwort fragen (interrupt)
    # Der Wert von interrupt(...) wird beim Resume die User-Antwort enthalten.
    company_reply = interrupt({
        "type": "company_response_required",
        "message": (
            "Formuliere deine Unternehmensantwort (Tweet/Statement/Posting) "
            "auf diese Kommentarwelle."
        ),
        "community_wave": comments
    })

    return {
        "community_reactions": new_log,
        "company_response": company_reply
    }


# -------------------------
# Node: Evaluation (Rubrik)
# -------------------------

def evaluation_node(state: SimState) -> SimState:
    cause = state["cause"]
    reply = state.get("company_response") or ""
    rep = state.get("reputation_score", 50)
    itr = int(state.get("iteration", 0))
    prev = "\n\n---\n".join(state.get("community_reactions", [])[-1:])

    rubric = (
        "Bewerte die Antwort des Unternehmens auf einer Skala 0-100 (0 sehr schlecht, 100 exzellent) "
        "nach diesen Kriterien: (1) Verantwortungsübernahme, (2) Empathie, (3) Konkrete Maßnahmen/Abhilfe, "
        "(4) Transparenz/Klarheit, (5) Ton (nicht defensiv, nicht relativierend). "
        "Gib ein JSON mit Feldern: score (int), label ('poor'|'mixed'|'good'), "
        "reasons (array kurzer Strings), suggestions (array kurzer Strings), "
        "resolved (bool, wenn Antwort realistisch beruhigend), "
        "catastrophe (bool, nur wenn Antwort die Lage massiv verschlimmert). "
        "Keine zusätzlichen Texte außerhalb des JSON."
    )

    prompt = f"""Anlass/Cause: {cause}

Jüngste Community-Reaktionen:
{prev if prev else "(keine)"}

Unternehmensantwort (zu bewerten):
\"\"\"{reply}\"\"\"

Wende die Rubrik an und antworte ausschließlich als JSON.
"""

    raw = llm.invoke(
        [
            {"role": "system", "content": rubric},
            {"role": "user", "content": prompt},
        ],
        # JSON-Modus (für OpenAI-Modelle unterstützt)
        # Fällt automatisch zurück, falls nicht verfügbar.
        # In vielen Setups funktioniert es ohne model_kwargs – daher minimal.
    ).content

    try:
        data = json.loads(raw)
    except Exception:
        # Fallback: naive Reparatur
        raw_fixed = raw.strip().split("```")[-1]
        try:
            data = json.loads(raw_fixed)
        except Exception:
            data = {
                "score": 40, "label": "mixed",
                "reasons": ["Konnte JSON nicht sauber parsen."],
                "suggestions": ["Klarere Entschuldigung", "Konkrete Maßnahmen nennen"],
                "resolved": False, "catastrophe": False
            }

    score = int(data.get("score", 50))
    label = str(data.get("label", "mixed"))
    resolved = bool(data.get("resolved", False))
    catastrophe = bool(data.get("catastrophe", False))

    # sehr einfache Heuristik um reputation_score zu aktualisieren
    delta = {"good": +20, "mixed": 0, "poor": -20}.get(label, 0)
    new_rep = _clamp(rep + delta)

    # Endbedingungen (simpel & transparent)
    status: Literal["ongoing", "resolved", "catastrophe"] = "ongoing"
    if resolved or new_rep >= 80:
        status = "resolved"
    elif catastrophe or new_rep <= 15 or itr >= 6 and new_rep < 60:
        status = "catastrophe"

    out = {
        "last_eval": data,
        "reputation_score": new_rep,
        "iteration": itr + 1,
        "status": status
    }

    return out


# ---------------------------------
# Graph aufbauen (Nodes + Kanten)
# ---------------------------------

def build_app():
    builder = StateGraph(SimState)

    builder.add_node("react", community_reaction_node)
    builder.add_node("evaluate", evaluation_node)

    builder.add_edge(START, "react")
    builder.add_edge("react", "evaluate")

    def continue_or_end(state: SimState):
        if state.get("status") in ("resolved", "catastrophe"):
            return END
        return "react"

    builder.add_conditional_edges("evaluate", continue_or_end)

    # Checkpointer für Interrupt/Resume (lokale SQLite-Datei)
    memory = SqliteSaver.from_conn_string("shitstorm_sim.sqlite")
    return builder.compile(checkpointer=memory)


# ---------------------------------
# Mini-CLI Runner (einfach gehalten)
# ---------------------------------

def run_cli():
    print("\n=== Shitstorm Simulation (LangGraph Minimal) ===\n")
    cause = input("Ursache/Anlass des Shitstorms (kurz beschreiben): ").strip()
    if not cause:
        print("Abbruch: Keine Ursache angegeben.")
        sys.exit(0)

    app = build_app()

    # Jede Simulation bekommt eine thread_id (damit Resume sauber funktioniert)
    thread_id = str(uuid.uuid4())

    # Initialer State
    initial_state: SimState = {
        "cause": cause,
        "community_reactions": [],
        "reputation_score": 50,
        "iteration": 0,
        "status": "ongoing",
    }

    # Erste Ausführung: streamen, Interrupt abfangen, dann resume
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": "main"}}

    # Einfache Event-Schleife: Wir streamen, prüfen auf Interrupts, fragen nach Antwort und resumen
    current_state = initial_state

    while True:
        # 1) Stream bis zum nächsten Interrupt / Ende
        interrupted = None
        checkpoint_id = None

        for event in app.stream(current_state, stream_mode="values", config=config):
            # Wenn ein Interrupt passiert, enthält das Event ein __interrupt__-Feld
            if isinstance(event, dict) and "__interrupt__" in event:
                interrupted = event["__interrupt__"]  # Liste von Interrupts
                checkpoint_id = event.get("checkpoint_id")
                break

            # Ausgabe nach Evaluierung (wenn vorhanden)
            if isinstance(event, dict):
                # Wenn gerade eine Evaluierung gelaufen ist, gib ein kurzes Feedback
                st = event
                if st.get("last_eval"):
                    le = st["last_eval"]
                    print("\n--- Bewertung (automatisch) ---")
                    print(f"Score: {le.get('score')} | Label: {le.get('label')}")
                    if le.get("reasons"):
                        print("Gründe:", "; ".join(le["reasons"]))
                    if le.get("suggestions"):
                        print("Vorschläge:", "; ".join(le["suggestions"]))
                    print(f"Reputation: {st.get('reputation_score')} | Runde: {st.get('iteration')} | Status: {st.get('status')}")
                    # Wenn beendet, Abschluss ausgeben und return
                    if st.get("status") == "resolved":
                        print("\n✅ Die Lage beruhigt sich. Shitstorm weitgehend abgeklungen.")
                        return
                    if st.get("status") == "catastrophe":
                        print("\n❌ Reputation stark beschädigt. Shitstorm eskaliert.")
                        return

        # 2) Falls interrupt: Community-Welle anzeigen, Unternehmensantwort abfragen, resume
        if interrupted:
            # Wir erwarten genau einen Interrupt pro Runde
            intr = interrupted[0]
            wave = intr.get("community_wave")
            if wave:
                print("\n=== Neue Community-Welle ===")
                print(wave)

            answer = input("\nDeine Unternehmensantwort: ").strip()
            if not answer:
                print("Hinweis: Leere Antwort eingegeben. (Das wird schlecht bewertet werden.)")

            # 3) Resume: Antwort an LangGraph übergeben, dann geht's weiter zur Evaluation
            if checkpoint_id is None:
                # Fallback, sollte nicht passieren – aber dann streamen wir einfach neu
                current_state["company_response"] = answer
                continue
            else:
                for event in app.resume([answer], checkpoint_id=checkpoint_id, stream_mode="values", config=config):
                    if isinstance(event, dict):
                        # Zustand auffrischen
                        current_state.update(event)
                # Danach läuft die Schleife erneut (Evaluation/Ende wird in der nächsten Iteration ausgegeben)
        else:
            # Kein Interrupt => Graph ist zu Ende gelaufen
            break

# neu rauf
# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    try:
        run_cli()
    except KeyboardInterrupt:
        print("\nAbgebrochen.")
