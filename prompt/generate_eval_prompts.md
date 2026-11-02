<!--
Usage: paste everything below the line into a NEW chat (not the one used for
calibration prompts). It returns 50 lines per reply; reply "continue" once to
get ~100 lines. Save them as eval_prompts.txt (one prompt per line) and use it
as EVAL_PROMPTS in the Kaggle notebook, or --prompts for
scripts/eval_prompts.py.

Keep this set separate from prompts.txt: evaluating on prompts the model was
calibrated on would overstate how well the pruned model holds up. To check
for accidental overlap:
    grep -Fxf prompts.txt eval_prompts.txt   # should print nothing
-->

---

I'm building a held-out evaluation set for a language model that writes product descriptions. Generate prompts that each ask for a product description, given a product name, its specifications, and a target audience. These prompts test how well the model generalizes, so they should be varied and somewhat more demanding than a typical training example.

## Output format (strict)
- Output **one prompt per line**, inside a single plain-text code block.
- Each prompt must be **a single line**: no line breaks, no numbering, no bullet points, no surrounding quotes.
- Every prompt must contain three labeled parts, in this order: `Name:`, `Specifications:`, `Target audience:`. An optional fourth part, `Requirements:`, may follow the target audience (see below).
- Every prompt must end with exactly: `Product description:`
- Produce **50 prompts per reply**. When I say "continue", produce 50 more that don't repeat any product, name, or audience used before.

Example lines (copy the format, not the content):
```
Write a product description for the following product. Name: Lumora Sunrise Alarm Lamp. Specifications: simulated sunrise over 10 to 60 minutes; 20 brightness levels; 8 nature sounds; FM radio; USB-C charging port; 17 cm diameter. Target audience: shift workers who struggle to wake up in dark winter mornings. Product description:
Draft an online store listing for this product. Name: Fernhollow Merino Base Layer. Specifications: 100% 200 gsm merino wool; flatlock seams; odor resistant; machine washable; sizes XS to XXL. Target audience: cold-weather commuters who cycle to work. Requirements: under 80 words, friendly but practical tone, end with a call to action. Product description:
```

## Product names
- Invent **fictional** brand and product names. Do not use real brands, trademarks, or real product names.
- Make names distinctive (e.g. unusual brand words) so they are unlikely to match products in any other dataset.

## Coverage
Half of the prompts should come from these everyday categories:
consumer electronics; kitchen appliances; clothing and footwear; beauty and personal care; outdoor and camping gear; pet supplies; smart home devices; sports equipment.

The other half should come from less common categories:
mobility and accessibility aids; laboratory and scientific equipment; professional audio and video gear; boating and marine accessories; beekeeping and homesteading supplies; collectibles and hobby kits; industrial safety equipment; aquarium and terrarium supplies; board game accessories; specialty tea and coffee equipment; vintage-style home goods; 3D printing supplies.

## Difficulty mix
- About **40%**: straightforward prompts like the first example (3 to 6 specs, no extra requirements).
- About **30%**: include a `Requirements:` part with one or two constraints, such as a word limit (50 to 150 words), a tone (playful, luxurious, technical, reassuring, minimalist), a format (a short paragraph followed by 3 bullet points), or "mention the warranty".
- About **20%**: dense or technical specs (7 to 10 items, model numbers, compatibility lists, certifications) that the description has to turn into plain-language benefits.
- About **10%**: tricky audiences, such as buyers purchasing a gift for someone else, non-expert parents buying for an expert child, a B2B purchasing manager, or older adults new to technology.

Also vary:
- **Instruction wording:** rotate between phrasings such as "Write a product description for…", "Draft an online store listing for…", "Write catalog copy for…", "Describe this product to a potential customer…", "Write the product page text for…". Don't use the same opening more than ~10% of the time.
- **Specifications:** separated by semicolons, with realistic units and numbers.
- **Target audiences:** specific phrases (age, expertise, budget, motivation, or needs), never just "everyone".
- **Length:** most prompts between 40 and 110 words.

## Don'ts
- Don't write the product descriptions themselves; only the prompts.
- Don't include harmful, regulated, or adult products (weapons, drugs, tobacco, alcohol, medical treatment claims).
- Don't add any text before or after the code block.
