

PYTHON 313 uv project

# EA forum analysis app

## Metrics one could generate:
* External validation score? falsometer?
   * Get LLM to check external sources and decide on a p(true) of each claim
* Pairwise conditional probabilities
   * Most load-bearing claim in doc?
   * Internal falsometer?
* Fringe score
   * What percentage of the people are likely to agree with this claim 1-100 (without context of post)
      * Could have EA-fringe and all humanity-fringe
* Persuasiveness score?
   * What percentage of the people are likely to agree with this claim 1-100 (*with* context of post)?
      * For EA forum readership and all humanity
* Novelty score?
   * What percentage of people are likely to have already considered this claim 
      * For EA forum readership and all humanity
* Inferential Support score
   * How much high quality supporting evidence or argument is provided
   * For claim or for post
* Value score
   * ???
   * For claim or for post
* Aura score
   * How Famous is the author of this post (for ea forum: 1-no search results, 100-will macaskill)
      * Websearch
      * Should this include internal fame? N posts?
   * Attempt to understand how much upvote signal explained just by fame?
   * 

## Unless otherwise stated compute all metrics 

## For an author:
* Analyze aggregated metrics (outlined above)
* Find author's most contradicted and most load bearing claims

## Things that could be 'interesting' to do with this data+metrics:
* Create leaderboards for claims
   * Create load-bearing claim leaderboard
   * Create contradicted claim leaderboard
   * Create a fringe-belief leaderboard
* Create leaderboards for posts
   * Most value?
* Create leaderboards for authors
* Correlate different metrics with upvotes
   * See which one explains upvotes the most (see if it's AURA).
* Create reviewer for ppl to paste their post into before publishing:
   * upvote predictor
      * create 'aura'-adjusted upvote predictor
      * Find most closely related posts
         * Posts that say the same things
         * Posts that would disagree
      * Highlight most {metric} claims in the post
         * Least externally validated
         * Most novel
         * Most fringe
         * Most load-bearing
         * Potentially contradictory
         * 

## What metrics we're to generate to start:
* One (usually) LLM call per metric or per group of related metrics per claim:
    * External validation of claim
        * Pass the claim into the model get score, allow websearching!
        * score 1-10 (1 - many reputable external sources imply this claim is false, 10 many reputable sources say it is true)
    * Obvious flaws in claim
        * Is there obvious feedback that would change what should be claimed here?
        * Step1 ask LLM for actionable feedback on the claim+post
        * Step2 ask other LLM how useful the actionable feedback is, how much better would the post be if it took it into account, scale 1-10 (2llm calls!)
    * Actionable
        * How actionable is this claim, score 1-10
        * Pass claim into model, get score
    * Novelty score?
        * What percentage of people are likely to have already considered this claim (then flip so 0 is low and 10 is high)
            * For EA forum readership
            * all humanity
        * Pass claim into model get score
    * Fringe score
        * pass claim into model, get score for what % of humanity would agree with this (estimate) score 1-10
            * EA readership
            * all humanity
    * Persuasiveness score
        * pass claim into model + fringe score + whole post text, get score for delta % that would change their mind on claim given the post
    * Inferential support score
        * pass claim into model + whole post text, get score 1-10 for how much (good) reasoning and evidence is provided in support of the claim

    * actually all of these should be scored 1-10, or converted to 1-10.
    * each metric will have a .py file
        * each file will have the pydantic model for the metric. this should include post_id and will often just have one other field {metric} and explantation, but sometimes will have more!
        * there will be a public fn that takes the post pyndantic model + claim (which we need a pydantic model for. should have claim_id, post_id, and claim: str) -> metric_pydantic_model
        * then we create claim_metric_registry.py with a function compute_metrics_for_claim(metrics: list[type[metrics], claim: Claim, post: Post]) for passed in type[metrics] and returns all the metrics for that post+claim



## Overview going to analyze posts on EA
TODO:
- ✅ create moreorlesswrong/claim_extractor.py to extract most salient claims from given a Post. returns a list[Claim]. should have one public fn. use old_example_claim_extractor.py for inspiration
- ✅ Create one metric file moreorlesswrong/metrics/novelty.py that defines a Novelty (pydantic) metric and provides the public fn for copmuting it from a Claim + Post
- ✅ Create a metric registry moreorlesswrong/claim_metric_registry.py that provides a central interface compute_metrics_for_claim
- ✅ Create a pipeline that takes a list of posts, list of metrics, and a version-id:
    - calls some claim extraction code, saves the claims. data/claims/{version_id}/{post_id}
    - calls metric generation code for each metric for each claim for a post then saves data/metrics/{version_id}/{post_id}
    - this should be interruptable! and if we add a new metric later it should just add those metrics, not rerun everything and not complete without doing anything
- ✅ Run the pipeline over get_representative_posts(n=10) with our Novelty metric
- ✅ Create a streamlit app that takes a version-id and a list of metrics, loads the saved metrics and plots distributions of those metrics, and scatter plots of those metrix X base_score
- ✅ Create metrics/inferential_support.py that evaluates how much reasoning and evidence supports each claim 

# Ideas for later maybe:
* Show top posts by metric
* Aggregate by author?
* Investigate which metrics are most correlated with upvotes
* Predict upvotes?

* feedback provider for post
    * what metric that explains upvotes are you missing
    * get top N upvoted previous posts. Compute pairwise conditional probs between these theses

