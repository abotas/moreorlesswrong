
# EA forum analysis app
py313 uv project

## What's implemented?
- A (not at all optimized) claim extractor. Extracts N most central claims
- A harness for generating metrics *on claims* (no cross-claim metrics for now)
   - to add a new metric implement a metrics/{metric}.py and add it to the @claim_metric_registry
- An interruptable/resumable pipeline that runs over a set of posts, extracts claims, generates metrics
- A streamlit app for seeing metrics across claims, and how they correlate with base_score
- **5 implemented metrics:**
   - **Novelty**: How new/original the claim is (EA audience & humanity)
   - **InferentialSupport**: Quality of reasoning/evidence supporting the claim  
   - **ExternalValidation**: Fact-checking against external sources
   - **Robustness**: Two-step evaluation of claim improvement potential
   - **AuthorAura**: Author fame/prominence scoring

## Issues 
- Might not be that useful without improving/droppingin replacement claim extractor
- I might be finding it overly tempting to ground in base_score. I think we can improve discourse by catching things that would score badly and help improve them before they're published, but their are often issues overly optimizing for upvotes

## TODO:
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
- ✅ Create metrics/external_validation.py that fact-checks claims against external sources
- ✅ Create metrics/robustness.py that evaluates claim improvement potential via two-step LLM feedback
- ✅ Create metrics/author_aura.py that scores author fame/prominence

## Usage:
Pipeline: `uv run python moreorlesswrong/pipeline.py --threads 4 --model gpt-5-mini`
Streamlit: `uv run streamlit run moreorlesswrong/streamlit_app.py v1` 

## Additional metrics to explore:
* One (usually) LLM call per metric or per group of related metrics per claim:
    * Actionable
        * How actionable is this claim, score 1-10
        * Pass claim into model, get score
    * Fringe score
        * pass claim into model, get score for what % of humanity would agree with this (estimate) score 1-10
            * EA readership
            * all humanity
    * Persuasiveness score
        * pass claim into model + fringe score + whole post text, get score for delta % that would change their mind on claim given the post

Metric afterthoughts
    * actually all of these should be scored 1-10, or converted to 1-10.
    * each metric will have a .py file
        * each file will have the pydantic model for the metric. this should include post_id and will often just have one other field {metric} and explantation, but sometimes will have more!
        * there will be a public fn that takes the post pyndantic model + claim (which we need a pydantic model for. should have claim_id, post_id, and claim: str) -> metric_pydantic_model
        * then we create claim_metric_registry.py with a function compute_metrics_for_claim(metrics: list[type[metrics], claim: Claim, post: Post]) for passed in type[metrics] and returns all the metrics for that post+claim

## Other metrics to generate

# Ideas for later maybe:
* generate cross claim or cross post or cross author metrics
   * conditional probs (higher N)
* feedback provider for post
    * highlight certain claims if high on metric scale, e.g. very load-bearing, very contradicted by external sources, etc
    * with owen's robustness metric set high cutoff and tell them feedback above that cutoff
      * could maybe eval this
        * by commenting on posts and see if comments get up/down votes
        * by backtesting and seeing if we catch ideas/sentiments that ppl comment?
    * explain what metric that correlates with upvotes are they missing
    * get top N upvoted previous posts. Compute pairwise conditional probs between these theses
    * Find most closely related posts
      * Posts that say the same things
      * Posts that would disagree
* or comment on some posts with this kind of info... maybe
* Create leaderboards for claims
   * Create load-bearing claim leaderboard
   * Create contradicted claim leaderboard
   * Create a fringe-belief leaderboard
* Create leaderboards for posts
   * Most value?
* Create leaderboards for authors
* Correlate different metrics with upvotes
   * See which one explains upvotes the most
