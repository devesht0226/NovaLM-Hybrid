# NovaLM-Hybrid Final Report

This document is the **written deliverable** alongside the runnable code: automated compare output (latency, retrieval samples) plus narrative sections. **Model metrics** below are merged from the previous report when present (otherwise defaults); **latencies** reflect the latest `run_compare.py` run. **Score columns** use `scripts/proxy_scores.py` heuristics unless you replace them via `scores.csv`.

## Run Summary
- Preprocessing: completed
- Tokenizer: completed
- Training: completed
- API: completed

## Metrics
- Validation loss (recomputed on `val_ids` from `token_ids.json`, `best.pt`): **6.5750**
- Validation perplexity: **716.94**
- Test loss (`test.txt`, tokenized): **6.5645**
- Test perplexity: **709.44**
- Avg latency (/generate): 348.45 ms
- Avg latency (/hybrid_generate): 1358.57 ms

## Prompt Comparison (Pure vs Hybrid)
| # | Prompt | Pure (short) | Hybrid (short) | P.coh | P.rel | P.grd | H.coh | H.rel | H.grd |
|---|--------|---------------|----------------|-------|-------|-------|-------|-------|-------|
| 1 | What is self-attention in transformers? | i have no, i have you. as not it is showed, o, and he was you must i know my lord, my bail say and i’ | had her, and to your sport; her, for your country’d. what you have i am i have you, of a light, that have his | 4 | 2 | 3 | 4 | 2 | 3 |
| 2 | Why do language models use tokenization? | to be my life. good my lord, for a captive. if i am me me. but i have i am not out for my life when you shall | with a plain a muse. the prince of the king. but when thou hast no, as much? in the duke, and my lord, and quickly, | 4 | 1 | 2 | 4 | 1 | 3 |
| 3 | Difference between top-k and top-p sampling? | to his hairy in the blood, i have i know not the while. what, or his face’d by the worth; in me her sleep, and | ’tis not to read thy death, now i’s court, for a palm and the day in the blood that i were it is of mine, and | 4 | 2 | 3 | 5 | 2 | 2 |
| 4 | What causes hallucinations in LMs? | with you. for these, the good will now, she must you. the way, i am me, this is, of death! what, and i | have done, if you shall not, i am not, with his injury. enter cassio and it. and he did plot the rich and with a true, | 4 | 2 | 3 | 4 | 1 | 2 |
| 5 | What is perplexity and why does it matter? | it, and, thou didst? in such a master. and our griefs, and others. so, for that he is be called he hopes to make. | that i will do’s an appetite. when i, to be not so, and i will not that it are a light, and not, of your | 4 | 2 | 3 | 4 | 2 | 3 |
| 6 | Write a short motivational message for students. | but for a pedant thee, i could-simple. o’s death, thou, ’tis to me to thy affairs, you shall the treasury. i | and others, but of the ward, in the better her, or, but a word, i have in that there’s the world have forgot and have | 5 | 2 | 4 | 4 | 2 | 2 |
| 7 | Explain recursion with a simple analogy. | i’s-k, that threatens the world you. if i am the gods, and yours. he is the doctor is the prince, with cap, | the world he were the ward, thou art if i will not, good, and yet it, it hath i’s nativity at my lord, i’ | 4 | 2 | 3 | 4 | 1 | 2 |
| 8 | Draft a polite email asking for project feedback. | now? the shoulder, he’d, and make the world, i am it is a favour to the king like by the place of the world to be | i am i would i am on the same for he ask. i cannot, and have that he provided, and yet there is the manner at your assis... | 4 | 2 | 3 | 4 | 2 | 2 |
| 9 | Give a 5-point plan to learn deep learning quickly. | enter that as much you, which; good my lord. but but i apply of force a palm, and with a word. a true’s death: | let me, the king. here, but the world with thy vocation; that is the duke to see a lu for the fool, enter cassio, and th... | 5 | 1 | 3 | 4 | 2 | 2 |
| 10 | Write a 4-line poem about coding at night. | the world’d. to my lord, and yet the time, i will like the duke of the view, if see and the worth of death, of | it had of her me, and make us, it will on the king. now. but i’s hoard, i not have gle me. for us | 4 | 1 | 2 | 4 | 1 | 3 |


*Heuristic score injection (scripts/proxy_scores.py): table scores are auto-filled. Edit `artifacts/reports/scores.csv` manually and run `python scripts/score_report.py` for rubric-only workflows.*

## Best Cases
1. **Retrieval plumbing:** TF-IDF returns ranked passages with stable scores (see **Retrieved Context Samples**); the hybrid path exercises encode-retrieve-decode without API errors.
2. **Pure latency:** The pure LM path stays faster when you only need a quick baseline sample (see **Metrics** latencies).

## Failure Case
- **Prompt:** Open-ended or technical prompts when the index is literary (e.g. Shakespeare chunks): hybrid answers read as garbled early-modern dialogue instead of the requested style (see rows #1-#10 in the latest compare run).
- **Issue:** Small LM capacity plus TF-IDF matching query tokens to unrelated but high-scoring passages; hybrid injects that context verbatim into decoding.
- **Likely fix:** Curate `data/raw/` closer to your task, tune chunking, add stopword filtering or a denser retriever, and train longer or with more model capacity.

## Retrieved Context Samples
- Prompt 1:
  - mine own self-love quite contrary i read: self, so self-loving were iniquity. ’tis thee, myself, that for myself i praise, painting my age with beauty of thy days. (score=0.1920)
  - you are a very simplicity ’oman; i pray you, peace.—what is _lapis_, william? a stone. and what is “a stone,” william? (score=0.1890)
  - brabantio appears above at a window. brabantio. what is the reason of this terrible summons? what is the matter there? (score=0.1762)
- Prompt 2:
  - i pray you, sir, deliver with more openness your answers to my demands. why do you pity me? that others do, (score=0.2173)
  - who hadst deserv’d more than a prison. you taught me language, and my profit on ’t is, i know how to curse. the red plague rid you, for learning me your language! (score=0.2127)
  - upon the right hand i. keep thou the left. why do you cross me in this exigent? i do not cross you; but i will do so. [_march._] (score=0.1971)
- Prompt 3:
  - i have dogs, my lord, will rouse the proudest panther in the chase, and climb the highest promontory top. and i have horse will follow where the game (score=0.2524)
  - we will, fair queen, up to the mountain’s top, and mark the musical confusion of hounds and echo in conjunction. i was with hercules and cadmus once, (score=0.2400)
  - the raven rooked her on the chimney’s top, and chatt’ring pies in dismal discord sung; thy mother felt more than a mother’s pain, and yet brought forth less than a mother’s hope, (score=0.2108)
- Prompt 4:
  - upon our spiritual convocation and in regard of causes now in hand, which i have opened to his grace at large, as touching france, to give a greater sum (score=0.2215)
  - his uncle siward, and the good macduff. revenges burn in them; for their dear causes would to the bleeding and the grim alarm excite the mortified man. (score=0.2061)
  - and youthful still—in your doublet and hose, this raw rheumatic day? there is reasons and causes for it. we are come to you to do a good office, master parson. fery well; what is it? (score=0.2057)
- Prompt 5:
  - then hamlet does it not, hamlet denies it. who does it, then? his madness. if’t be so, hamlet is of the faction that is wrong’d; his madness is poor hamlet’s enemy. (score=0.2487)
  - music i’ th’ air. under the earth. fourth soldier. it signs well, does it not? (score=0.2465)
  - o heavens, why does my blood thus muster to my heart, making both it unable for itself and dispossessing all my other parts (score=0.2155)
- Prompt 6:
  - your message done, hie home unto my chamber, where thou shalt find me sad and solitary. how many women would do such a message? alas, poor proteus, thou hast entertained (score=0.2365)
  - he hath some message to deliver us. ay, some mad message from his mad grandfather. my lords, with all the humbleness i may, i greet your honours from andronicus; (score=0.2325)
  - unless her prayers, whom heaven delights to hear and loves to grant, reprieve him from the wrath of greatest justice. write, write, rynaldo, to this unworthy husband of his wife; (score=0.1613)
- Prompt 7:
  - enter simple. how now, simple, where have you been? i must wait on myself, must i? you have not the _book of riddles_ about you, have you? simple. (score=0.4824)
  - simple of my life for an hour and a quarter. the fee simple! o simple! enter tybalt and others. by my head, here comes the capulets. (score=0.4447)
  - [_aside_.] if a talent be a claw, look how he claws him with a talent. this is a gift that i have, simple, simple; a foolish extravagant spirit, full of forms, figures, shapes, objects, ideas, apprehensions, motions, revolutions. these are begot in the ventricle of memory, (score=0.2881)
- Prompt 8:
  - a labour sav’d! a wonder! ajax goes up and down the field asking for himself. how so? (score=0.2176)
  - ’tis the cardinal; and merely to revenge him on the emperor for not bestowing on him at his asking, the archbishopric of toledo this is purposed. (score=0.1739)
  - come from the north: and as i came along, i met and overtook a dozen captains, bareheaded, sweating, knocking at the taverns, and asking everyone for sir john falstaff. (score=0.1650)
- Prompt 9:
  - which, though i will not practise to deceive, yet, to avoid deceit, i mean to learn; for it shall strew the footsteps of my rising. but who comes in such haste in riding-robes? (score=0.1666)
  - by any means; our thing of learning says so— where he himself will edify the duke most parlously in our behalfs. he’s excellent i’ th’ woods; bring him to th’ plains, his learning makes no cry. (score=0.1660)
  - to learn his wit t’ exchange the bad for better. fie, fie, unreverend tongue, to call her bad whose sovereignty so oft thou hast preferred with twenty thousand soul-confirming oaths. (score=0.1642)
- Prompt 10:
  - rugby. i’ll go watch. go; and we’ll have a posset for’t soon at night, in faith, at the latter end of a sea-coal fire. (score=0.1999)
  - the cuckold’s horns. master brook, thou shalt know i will predominate over the peasant, and thou shalt lie with his wife. come to me soon at night. ford’s a knave, and i will aggravate his style. thou, master brook, shalt know him for knave and cuckold. come to me soon at night. (score=0.1939)
  - go in, and cheer the town; we’ll forth, and fight, do deeds worth praise and tell you them at night. farewell. the gods with safety stand about thee! [_exeunt severally priam and hector. alarums._] (score=0.1831)

## Score Averages
- Pure coherence: 4.20
- Pure relevance: 1.70
- Pure groundedness: 2.90
- Pure overall: 2.93
- Hybrid coherence: 4.10
- Hybrid relevance: 1.60
- Hybrid groundedness: 2.40
- Hybrid overall: 2.70
- Winner: pure
