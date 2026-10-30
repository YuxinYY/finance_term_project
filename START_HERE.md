# START HERE — using this kit with Claude Code in VS Code

## 1. Install (5 minutes)
1. Copy everything in this folder into your project repo root (keep the hidden `.claude/` folder).
2. Put Roger's data files into `data/raw/` (train.csv and any ATECO definition doc).
3. `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
4. `python -m pytest src/common -q` → should print `3 passed`.
5. Open the folder in VS Code and start Claude Code there. It reads `CLAUDE.md` automatically
   and sees the 12 agents in `.claude/agents/` (check with `/agents`).

## 2. What's in the box
| Path | What it is |
|---|---|
| `CLAUDE.md` | The project brief + hard rules every session follows |
| `.claude/agents/*.md` | 12 specialised agents: 4 competing modellers, a referee, a red team, etc. |
| `.claude/commands/` | `/start-session`, `/tournament-round`, `/log-decision` |
| `docs/PLAYBOOK.md` | Phase plan to Nov 16, tournament rules, hypotheses to test |
| `docs/FEEDBACK_RESPONSE.md` | Every comment Roger made → the fix and the owning agent |
| `docs/DECISION_LOG.md` | Where every choice (adopted or rejected) is defended |
| `docs/REFERENCES.md` | Bibliography with verification status — nothing gets cited unverified |
| `docs/REPORT_TEMPLATE.md` | Deck structure in the order the assignment asks |
| `src/common/` | Shared label + metrics code (tested) — the referee's single source of truth |
| `harness/harness.py` | Harness skeleton matching Appendix A |

## 3. How the competition works
Four challengers build models under identical rules (same splits, same features, same output
file format). They do **not** grade themselves: the referee scores everyone with the same code,
compares them with firm-clustered bootstrap confidence intervals, and only promotes a model that
is better on probability accuracy or power **without** breaking face validity. Ties go to the
more explainable model. The red team then attacks the result the way Roger attacked the proposal.

```
data-auditor ─┐
              ├─> feature-architect ─> [riskcalc | scorecard | gbm | gam] ─> referee ─> red-team
lit-scout ────┘                                         ▲                       │
                                                        └──── next round ◄──────┘
                    calibration-specialist ─> harness-engineer ─> report-writer
```

## 4. Prompts to paste, in order
1. `/start-session`
2. "Run Phase 0: check data/raw, and draft the email to Roger/TA with the four questions in the
   PLAYBOOK. Don't send anything." (Then send it yourselves.)
3. "Run Phase 1: launch data-auditor and literature-scout in parallel."
4. "Run Phase 2: launch feature-architect, then red-team-roger on the feature spec."
5. `/tournament-round 1 baselines and first versions of each challenger`
6. `/tournament-round 2 missing-data treatments and feature variants`
7. "Run the calibration round: calibration-specialist on the top two, then referee."
8. "Launch harness-engineer on the winning model; run all harness tests."
9. "Launch report-writer, then literature-scout claim audit, then red-team-roger on the deck."

## 5. Things only you (the humans) can do
- Answer the questions agents mark OPEN (especially the label window — ask Roger).
- Read the core papers yourselves (at least RiskCalc v3.1 and Stein 2005/2007). Roger grades
  whether **every member understands the whole project**, and `explain_back.md` is your rehearsal.
- Write the contributions appendix truthfully and review the AI-use statement.
- Sanity-check what the model does on real examples before submitting (the assignment begs you to).
