# Speaker notes: Energy-Aware Agentic AI (about 18 minutes)

Each overlay step (a "click") is marked ▶. The slides carry only the key fact; the sentences below are what you say.

## Introduction (2 min)
**One prompt is cheap. An agent is not.**
▶ "One chatbot message costs about 0.6 watt-hours, less than ten seconds of a microwave. So why worry?"
▶ "Because there are billions of them: data centres already use 485 TWh a year, heading for 950 by 2030."
▶ "And agents multiply calls: on SWE-bench, two leading agents made 28 and 51 model calls for a single task."
▶ Read the Agent definition, pointing at generate / check / decide.

**The question of this seminar.** Trace the loop with your hand: task, generate, check, decide. ▶ Land the question: is another step worth its energy, how should we spend it, and when do we stop?

## Foundations (3 min)
**How a model answers.** Tokens, one at a time; temperature. ▶ "Every token is a pass through the network, so energy follows tokens, including hidden reasoning. Remember this for the last section."

**Ways to spend an extra call.** Walk down the table; the two rows that matter are fresh retry and repair with feedback. Point at the green box: independent attempts would give 75% from two 50% tries, "but hard problems are not coin flips".

**Collision problem (interactive).** Read the problem and ask the room in your head: "n cars each way: how many collisions?" Pause.
▶ Show the model's answer `return n`. ▶ "The test with n = 1 passes, because 1 equals 1 squared." ▶ "n = 2 fails." ▶ "Given the failure, it fixes it: n times n." ▶ Twist: `return 4` would also pass. "Tests the agent sees are not proof, so I score everything on hidden tests."

## Measuring energy (2 min)
**Power to energy.** Point at the two spikes. ▶ "31.5 J for the wrong answer plus 27.6 J for the fix: 59 J for one correct answer. The failure is part of the price."

**Energy per correct solution.** ▶ Break-even inequality. ▶ Say the rule in plain words: 10% more energy must buy more than 10% more success. ▶ Emissions factor (Indian grid).

## Experiments (6 min)
**Setup.** Laptop GPU, two model sizes, 140 problems. Use the diagram: "every policy starts from the same failed program, so differences are policies, not luck."

**Pilot.** ▶ Energy +44–52%, success +3–5%. ▶ "13 runs used all four attempts and still failed: nearly 40% of the energy bought nothing."

**Surprise #1 (interactive).** Ask: "When the model sees its own error message, how often do you think it changes its code?" Let it hang. ▶ Chart appears. ▶ "76 to 91% of the time it sends back exactly the same program." ▶ So short vs full history makes no difference.

**New experiment: is it just the low temperature?** Raise the objection yourself: "Maybe the model can't change because I set temperature to 0.2." ▶ "So I reran every repair at 0.8, four times hotter, from the same failed answers: still 85% and 72% unchanged." ▶ "And it fixed no more problems: 2 and 2 for the small model, 1 and 2 for the large one." ▶ "It isn't stuck on randomness. It's confident in its wrong answer."

**Live demo: repair vs. fresh retry.** Run it live if you can (commands below). Left box: "Repair: the same program four times, 9 of 10 tests each time, 372 joules, still wrong." ▶ Right box: "A fresh retry: solved on the second try, 173 joules." ▶ "The numbers change a little on each run; the pattern doesn't." 

**Surprise #2.** "Starting over beats fixing": fresh retry rescued 12 of 42 failures for the small model. ▶ But no multi-attempt policy beat one attempt on energy. ▶ Lesson box.

**When to stop.** Define the "no progress" rule. ▶ Saves a quarter to a third of energy... ▶ "...but it's just a cap of two in disguise." ▶ Wrong for retries. ▶ Takeaway.

**Small model + retries.** Ask: "Can a model five times smaller catch up by retrying?" ▶ 75.0% vs 79.3% ▶ at 0.83× the energy. ▶ Not statistically clear; a serious alternative, not a winner.

## Frontier-scale models (3 min)
**Scaling up.** "I ran the same 140 problems on Kimi K3.1 Pro, 2.8 trillion parameters, and Nemotron 3 Ultra, 550 billion." 93% first time; retries cannot close an 18-point gap. ▶ Switch to the token chart: Nemotron thinks 536 tokens before answering; the whole Kimi run cost 46 cents.

**Surprise #3.** "Bigger models fooled the checker more often, not less." ▶ Only better verifiers help. ▶ Energy you cannot see: Google's 0.24 Wh median vs my 152 J; API users see tokens and prices, never energy.

## Conclusion (2 min)
**Policy.** India's guidelines; the three-line disclosure card filled in for the collision run.
**Summary.** Read the four bullets slowly; end on "Not answered" to show you know the limits.

## Live demo setup (laptop, experiment folder)
1. Start Ollama (`ollama serve`) and check the NVIDIA GPU is active.
2. `python live_demo.py --task HumanEval/134` — repair; press Enter for each attempt. It prints the code, the visible-test result and the joules used.
3. `python live_demo.py --task HumanEval/134 --mode retry` — fresh retry for contrast.
4. `python live_demo.py` (no task) runs the collision problem; Qwen 1.5B usually solves it first time, which makes a quick warm-up.
The demo slide already shows the rehearsal results, so if anything fails on stage, just present the slide.
