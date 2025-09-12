Epistemic Virtue for AI Models
Tl;dr: This is a document outlining the different ways in which an agent or a system can be epistemically virtuous for purposes of epistemic evals. It presents three primary virtues--Accuracy, Value, and Cooperativeness--and some derivative virtues that contribute to the primary virtues.
Introduction
A crucial component of the current AI ecosystem are evals, short for evaluations. These are testable metrics designed to assess how well a system performs in some domain. One domain in which we can assess a system is how it performs epistemically. Call a system that performs well in this way epistemically virtuous. 

In this document, we attempt to make clear what it takes for a system to be epistemically virtuous, and by extension what epistemic evals should ultimately be aiming to assess. I propose three primary virtues that constitute the epistemic virtue of an agent:

Accuracy - “Is the system providing accurate information?”
Value - “Would the information provided make a difference if it were accurate?”
Cooperativeness - “Is the system providing information in a way that improves the epistemic situation of the recipient?”

These are mutually irreducible and individually necessary, meaning a system can perform well on one without performing well on the others, and that failing on any one of them will undermine epistemic virtue. To illustrate, here are three respective example failure modes that can arise if one of them is missing:

No Accuracy: A system provides clear and impactful information that is completely false.
No Value: A system says a series of true platitudes instead of providing important information.
No Cooperativeness: A system provides true and important information in an invented language that can only be decoded at great cost.

I contend that they are also collectively sufficient: if an agent performs well on all three, then they are epistemically virtuous.

I also present a list of derivative virtues. These are properties and capabilities that also contribute to a system’s overall epistemic virtue. However, unlike the primary virtues, they are important only because they contribute to a system’s performance on one or more of the primary virtues. Here is a list of the derivative virtues, and an overview of the primary virtues they contribute to:



Accuracy
Value
Cooperativeness
Coherence
✔️




Attunement
✔️




Creativity
✔️
✔️


Activity


✔️
✔️
Clarity


✔️
✔️
Explainability




✔️
Causal transparency




✔️
Precision
✔️
✔️
✔️
Honesty
✔️
✔️
✔️



The goal of this document is not to argue that epistemic evals should aim to specifically test the dimensions proposed here. For example, a test for sycophancy might be very useful because it implicitly demonstrates that a model fails several virtues. Rather, the goal is to provide a structured overview and guide to what it is we should care about when we design epistemic evals, and to explain why models that fail such evals are failing in a normatively important way.

In the next section, I present the primary virtues in more detail. In the section that follows, I present the derivative virtues in more detail. In the final section, I consider how they apply to some use cases.
Primary Virtues
Accuracy
“Is the system providing accurate information?”

Accuracy is about whether the statements of the system correspond to how things are: whether they are true. For example, a system that frequently spouts falsehoods will score low on Accuracy. Accuracy is arguably the most obvious dimension of epistemic virtue. Making true claims and not making false claims is essential to being epistemically virtuous.

A natural way to represent Accuracy is to use scoring function. A very simple one is to give an answer 1 if it is true, and 0 if it is false. A system’s Accuracy can then be the percentage of answers that are true out of all answers given.

If the system provides probabilistic estimates, then this can be measured with a probabilistic scoring rule such as a Brier score. With these scores, a System is rewarded for giving probabilistic estimates closer to 1 when the claim is true, and closer to 0 when the claim is false.

Sometimes we weigh saying true things less highly than we weigh not saying false things. This can be represented by introducing weights in scoring rules, where a model is punished more for saying something false than for not saying something true.

A system can be Accurate in more complicated ways. When the question is about a quantity (e.g. “How many days is the gestation period of an African elephant?”), a virtuous agent with imperfect knowledge provides a probability distribution that is appropriately spread out across answers, being neither overconfident nor underconfident in their answer.
Value
“Would the information provided make a difference if it were accurate?”

One way to be maximally Accurate is to say one true thing and then never say anything again. By contrast, virtuous epistemic agents say things that matter and aim to communicate valuable information. This is what we aim to capture under the virtue of Value.

One way to conceptualize Value is whether the information provided would make a difference to an agent. This question can be broken down further into two questions:

Does the information make a difference to an inquiry about the world?
Does the inquiry make a difference to an agent?

Suppose I have a party tonight and I receive information that there will be a storm starting at 9pm. This information makes a difference to the inquiry of whether there will be a storm by ruling out the prospect--or the possible worlds, as philosophers say--in which it doesn’t rain tonight. Furthermore, this inquiry makes a difference to me, because it determines whether I bring an umbrella. This information has high Value.

Contrast this with two bad cases. First, suppose instead that I receive information that water is wet. This makes no difference to any realistic inquiry, and so it has low Value. Second, suppose I receive information that David Hume died at 3.51pm. This information makes a difference to several inquiries (e.g. “What time did Hume die?”), but it makes not to any inquiry that matters to me given my interests, and so it too has low Value.

Value can be represented in different ways. The informativeness of some piece of information--how much difference it makes to some inquiries--can be represented in terms of Shannon information. Intuitively, this is a measure of how many possible worlds the information rules out. For example, the claim that Hume died at 3.51pm is extremely informative, because it rules out that he died at any other time.

This only captures the first question, however. To also represent the value of an inquiry, one needs to introduce a representation of what matters to an agent, such as a utility function. We can then represent the value of information as how informative the information is to the inquiry, weighted by how big a difference that inquiry makes to decisions that the agent can take, given their utility function. Intuitively, this is a measure of what the agent would pay to receive that information.
Cooperativeness
“Is the system providing information in a way that improves the epistemic situation of the recipient?”

A virtuous epistemic does not only provide Accurate and Valuable information. They also take responsibility for how to convey the right information in a way that is beneficial to the recipient, putting them in the best epistemic situation they can. This can be thought of as effectively causing appropriate understanding in an intended subject. Cooperativeness is the virtue of doing this well.

Being cooperative involves being intelligible, explanatory, and epistemically aligned. Being intelligible means presenting information in a way that is accessible to the recipient at low cost. For example, in explaining Newtonian physics, a cooperative agent will do so differently to a 5-year-old, a high school student, and a college student majoring in physics.

Being explanatory means providing the context that provides the recipient with the autonomy to assess an answer and related inquiries themselves. This comes in two different categories. One is explaining the inferential support of the conclusion that the system generates so that a recipient can assess it for themselves. The other is being transparent about the causal processes that generate the output. These can be thought of as the justification of an answer and the algorithm that generates the answer, respectively.

Finally, a system can be both intelligible and explanatory yet fail to be cooperative if it is not epistemically aligned. Specifically, it might selectively provide information in a manner that is manipulative, putting the recipient in a worse epistemic situation even though the information is accurate, valuable, and clearly conveyed and explained. An example of a system that fails in this way is media outlets that face incentives to grab attention. They might provide disproportional reporting on outlandish events like terror attacks, causing inaccurate beliefs about the prevalence of those events.

Representing Cooperativeness is more amorphous than Accuracy and Value, because it is a matter of judgment by the recipient and observers. However, in principle, it can be measured by seeing how interactions with the system affect participants and whether those interactions tend to bring about better understanding.
Derivative Virtues
The primary virtues are properties that we want a system to have for its own sake. However, there are several other properties that we think a system should demonstrate. I categorize these as derivative virtues. They are important because they contribute to a system having the primary virtues. Here I consider some salient ones, while leaving open that there might be many more.
Coherence
“Are the claims expressed by the agent mutually consistent?”

Accuracy
Value
Cooperativeness
✔️






Coherence is the virtue of having one's attitudes be mutually compatible. For example, a system that expresses both the propositions p and ~p is expressing incompatible attitudes. In many cases, one’s attitudes are less obviously incompatible, however.

For credences, a standard test for problematic incoherence is whether the views sanction buying a Dutch Book, which is a set of bets that jointly guarantee a loss. For example, if I bet on a race between A, B, and C (no ties), and I believe their chances of winning are 20%, 40%, and 50% respectively (adding up to 110%), and buy bets corresponding to those probabilities, then I will necessarily have overpaid and lost money. (This is always true when one’s credences violates the probability axioms).

Coherence comes in two forms. Synchronic and diachronic incoherence, meaning incoherence at a given time vs. across time. Arguably, diachronic incoherence is not a problem, because it might just mean that you have changed your mind on good evidence. However, synchronic inchoerence seems to suggest a problem.

Being incoherent implies that you are inaccurate in some way. If you believe a set of mutually incompatible propositions, then at least one of them must be false. This is a reason to strive for Coherence. However, one can be coherent yet be inaccurate. For example, imagine a conspiracy theorist who is perfectly coherent in a fundamentally inaccurate worldview: everything “fits together”. 

Some would argue that Coherence is a distinct primary epistemic virtue. I contend that it is not, but that it is important insofar as it undermines Accuracy.
Attunement (Appropriate Robustness)
“How easily does the system express something different? Does this change track the evidence?”

Accuracy
Value
Cooperativeness
✔️






Attunement is the virtue of being appropriately sensitive to the evidence. One way to fail at being attuned is to change one’s view because of irrelevant influences. Another way is to not change one’s view at all when new evidence comes in.

Being attuned is important because responding appropriately to the evidence is necessary to have Accurate attitudes. At the same time, it is not a primary virtue, because we don’t care whether a system is responsive to evidence if it never translates to better behavior.

On an abstract level, a perfectly attuned system will update its attitudes following Bayes’ theorem. In practice, evaluating whether a system does is difficult because it depends on what implicit attitudes and conditional probabilities it might have possessed.

In practice, testing Attunement will likely involve stress-testing whether a system will change its attitude as a function of irrelevant influences, and conversely whether it will update its answers in response to new relevant information.
Creativity
“Can the system generate novel kinds of information when appropriate?”

Accuracy
Value
Cooperativeness
✔️
✔️




Some systems can only generate answers within a very well-known domain. However, many questions have answers that we haven’t yet imagined. Providing novel and correct answers to such questions is crucial to being Accurate in original domains. Additionally, such questions are typically especially valuable to answer, making this an important capacity for a system to demonstrate Value.

Some examples of systems that demonstrate Creativity might include scientists who come up with novel theories that are not merely extrapolations of previous paradigms (e.g. Einstein’s subsuming Newtonian physics). Another example might be an entrepreneur inventing new products that people did not know they wanted. Some artificial systems have arguably demonstrated Creativity in generating novel proofs in math or strategies in Go, for example.
Activity
“Does the system actively provide new information?”

Accuracy
Value
Cooperativeness


✔️
✔️


To create epistemic Value of time, a system has to actively provide new information that contributes to an agent’s epistemic position. For example, an agent could be perfectly Accurate over time by saying one truth and then never speaking again, yet this would limit how much difference-making Value the system can generate.

One way to frame Activity is that the system provides information for free that a recipient would be willing to pay for. The more such information it actively provides, the more Value it contributes.

Being Active can also be a way of being Cooperative. Some recipients might benefit from not having to themselves ask for the information to be provided. A Cooperative agent actively provides the information that the recipient needs without requiring the cost of asking for it, though without additionally providing superfluous information that the recipient did not need.
Clarity
“How easy is it to understand the information that the system provides?”

Accuracy
Value
Cooperativeness


✔️
✔️


The Clarity of a communicated message is how costly it is for the intended recipient to extract the information from the message. An epistemic system that performs well on Clarity provides messages that require low cost for processing by the recipients. This is a crucial part of being a Cooperative agent.

We can communicate messages in very complicated ways, even if they carry the exact same information. For example, the sentences ‘The number of sheep is the twelfth prime number’ and ‘The number of sheep is 37’ express the same content, but one is costlier to understand. This performs worse on Clarity.

Clarity can trade off against the Precision--and by extension the Value--of a message. For example, when explaining something to a young child, you might have to omit details that would be valuable if the child could understand it. Being an epistemically virtuous agent requires balancing these considerations by taking into account the cognitive budget of the recipient to process the messages provided.
Explainability
“Does the system explain why the information provided is correct, and help the recipient understand it?”

Accuracy
Value
Cooperativeness




✔️


A system that only provides answers is not maximally helpful, even if the answers are accurate. This is because the recipient won’t be able to assess or make use of those answers without further context, making the system lack in Cooperativeness. Performing well on Explainability means providing an explanation of an answer that rationalizes why that answer is correct, when such an explanation is beneficial to the recipient.

Examples where Explainability is important include high-stakes decision-making involving humans, such as healthcare decisions. If an automated system proposes a treatment for a patient, it is important to be able to provide an explanation for why this treatment is expected to be effective.

Explainability is important for two reasons. First, it allows the recipient to verify the veridicality of the information more easily. Second, it provides the recipient with autonomy to make flexible judgments using the information, without relying on the system entirely.
Causal Transparency
“Is the causal explanation of why a system generated a particular piece of information transparent to an observer?”

Accuracy
Value
Cooperativeness




✔️


A system might provide a justification for a judgment and thereby explain why the judgment is true. However, that justification might have nothing to do with the causal explanation of why the system generated the judgment. Ideally, a system will be transparent in its internal process, allowing observers to see why it arrives at the answers it does. When it does, it performs well on Causal Transparency.

To see how Causal Transparency differs from Explainability, consider the medical case again. Even if a system provides a good justification for why a particular treatment is effective, the causal explanation of why the system arrived at that treatment proposal might be unrelated to that justification. For example, perhaps it generated the judgment on the basis of a statistical correlation to the symptoms, and only generated an explanation of why the treatment is effective post hoc after the judgment.

Causal Transparency contributes to Cooperativeness because it allows an independent observer to understand whether the information provided is plausibly generated by a reliable process. It also allows them to verify whether the system is epistemically aligned with the recipient, or whether there is a risk of manipulation.

Interestingly, Causal Transparency is arguably one epistemic virtue that humans severely lack. Contrary to intellectualist explanations of why we believe what we do, it seems that human judgments are often generated by opaque processes, the output of which are subsequently justified with explicit explanations.
Precision
“How informative are the claims made by the system?”

Accuracy
Value
Cooperativeness
✔️
✔️
✔️


Precision is a measure of how informative the claims of a system are, in the sense of Shannon information. Put differently, the higher precision, the more possible ways the world could be that the claim rules out. 

One way to be maximally Accurate is to say things that are uncontroversially true, which are therefore very uninformative. However, such claims will be of low Value, because they make no difference to any inquiry. Being appropriately Precise requires saying the most informative thing you can that brings marginal value, without sacrificing too much Accuracy. Additionally, adding just the right amount of information is important to be Cooperative with recipients with limited information-processing capacities.

One way to test appropriate Precision is to use confidence intervals (or probability distributions) over quantity estimates. For example, we might ask a system: What is the narrowest interval at which you are 80% confident that the correct answer will be within this interval?

A system’s output can also be imprecise by being ambiguous. For example, the statement ‘That book is pretty heavy’ can be interpreted either as saying that the book is heavy reading, or that it’s heavy to carry around. Typically, this is determined by pragmatic context, e.g. whether someone is considering travelling with the book. The fewer alternative readings that an expression allows for, the higher the informativeness of that statement.

Although more precise information typically increases Value, sometimes that information only makes a difference to inquiries we don’t care about. For example, if I am choosing college, I don’t need to be informed about the building materials of the registrar’s office. Adding superfluous information of this kind increases the risk of being inaccurate at no cost, and clutters the processing of the recipient, which makes it less Cooperative.
Honesty
“Is the system optimizing for a goal that is distinct from what the recipient believes the system is optimizing for?”

Accuracy
Value
Cooperativeness
✔️
✔️
✔️


In general, performing well on any metric of epistemic virtue in the limit requires that the system has as a goal to perform well on that metric. A deceptive system might overtly try to perform well on a metric being tested, while in fact performing poorly in a way that is being neglected.

An example of such deceptive behavior is providing selective information that is itself accurate, but that causes an inaccurate representation of the world in the recipient. For example, a model might provide selective research reports on a topic supporting only one side. This information might be valuable, but not as valuable as a more complete picture would be.

Lacking Honesty is most clearly a deficiency of Cooperativeness, as it is likely to put the recipient in a worse epistemic position. However, in practice, it is likely that a deceptive system will provide information in a way that lacks in both Accuracy and Value, relative to a system that has high Honesty. For this reason, it bears on all three primary virtues.
Case Study: Sycophantic Agent
[To be added]
