# Solution Summary — AI Air Travel Companion

**Participant:** Jay Balasaheb Mali · ch24b067@smail.iitm.ac.in

## 1. Selected problem statement

**AI Air Travel Companion** — an AI-powered air travel assistant that suggests
and optimizes flight options based on user profiles and inferred travel
preferences, supporting direct-vs-connecting trade-offs, flexible dates,
multi-city journeys, and seasonal pricing awareness.

## 2. The user / business problem

Flight search treats every traveler identically: a broke student and a
business-class executive see the same ranked list and must manually translate
their own priorities into filters. Preferences that travelers _express_ — in
reviews, notes, support chats ("redeyes kill my mornings", "cheapest only, 2
stops fine") — are never used. And when rankings do get personalized, they are
opaque: the user cannot see _why_ option A beat option B, which erodes trust
and conversion. For Expedia, unexplainable personalization is both a product
gap and a trust risk.

## 3. The proposed solution

A Streamlit prototype that closes the loop from messy signal to explained
recommendation:

- **Infers preference weights** (budget / time / comfort) by combining
  structured profile fields with LLM analysis of unstructured `raw_history`.
- **Understands plain-language requests** — the official benchmark prompts can
  be typed verbatim; an LLM extracts origin ("home" → home airport),
  destination (including region-level asks like "Asia"), trip type, date
  windows, and total trip duration.
- **Routes over a time-dependent flight graph** — direct itineraries plus
  synthesized connections that are only valid if the second leg departs 45
  minutes to the user's max layover after the first arrives.
- **Ranks by an explainable dollar model** — weights become an implicit
  Value of Time ($/hr); flights are ranked by
  _effective cost = price + VOT × hours + stop penalty_. The same route
  yields opposite recommendations for a student ($7/hr) and an executive
  ($210/hr), and the engine can justify each in one auditable sentence.
- **Plans round trips and multi-city journeys** with time-feasible leg
  chaining, duration budgets ("about three weeks"), and stay allocation.
- **Degrades gracefully and honestly** — when hard constraints eliminate every
  option, the engine relaxes the layover cap while keeping the user's dates,
  apologizes, and visibly badges each flight that exceeds their stated
  preference. Empty date windows trigger an explicit offer ("found 4 options
  in the following month — search that window?") rather than silently
  searching other dates. Engine-built self-transfer connections are labeled
  as separately ticketed.
- **Explains every pick** — an LLM insight grounded strictly in the real
  computed alternatives, plus an explicit Cheapest / Fastest / Our-pick
  trade-off strip and a price-vs-duration chart.

Built with the toolkit's suggested stack: Python, Streamlit, Pandas, Plotly,
and an open-source LLM (Llama 3.3 70B).

## 4. Expected value / impact

- **Higher conversion through trust:** recommendations justified in dollars
  ("$380 saved is outweighed by 292 minutes valued at $1,033 for you") give
  travelers a reason to accept the top result instead of re-sorting by price.
- **Latent-signal monetization:** preference text that currently sits unused
  in reviews and support logs becomes a personalization input with no extra
  user effort.
- **Fewer dead ends:** transparent constraint relaxation converts "no results"
  moments — a known abandonment point — into honest, bookable alternatives.
- **Extensible core:** the effective-cost formulation gives the business a
  single interpretable lever (VOT) that can later be learned per user from
  booking outcomes, turning a hackathon heuristic into a production
  personalization signal.
