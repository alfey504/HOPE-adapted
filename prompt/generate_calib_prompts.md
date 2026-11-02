<!--
Usage: paste everything below the line into a chat LLM. It returns 50 lines per
reply; reply "continue" until you have ~300 lines. Save them as prompts.txt
(one prompt per line) and use it as the calibration set (CALIB_PROMPTS in the
Kaggle notebook, or --prompts for `hope calibrate`).
-->

---

I'm building a calibration dataset for a language model that writes product descriptions. Generate prompts that each ask for a product description, given a product name, its specifications, and a target audience.

## Output format (strict)
- Output **one prompt per line**, inside a single plain-text code block.
- Each prompt must be **a single line**: no line breaks, no numbering, no bullet points, no surrounding quotes.
- Every prompt must contain three labeled parts, in this order: `Name:`, `Specifications:`, `Target audience:`.
- Every prompt must end with exactly: `Product description:`
- Produce **50 prompts per reply**. When I say "continue", produce 50 more that don't repeat any product, name, or audience used before.

Example lines (copy the format, not the content):
```
Write a product description for the following product. Name: AeroBrew Pro 12-Cup Coffee Maker. Specifications: 1.8 L glass carafe; 1,100 W heating element; programmable 24-hour timer; auto shut-off after 2 hours; reusable mesh filter. Target audience: busy professionals who want fresh coffee ready when they wake up. Product description:
Create marketing copy for this product. Name: TrailMate 40L Hiking Backpack. Specifications: 40 L capacity; 1.2 kg; water-resistant ripstop nylon; ventilated back panel; hydration sleeve; lifetime warranty. Target audience: weekend hikers planning their first multi-day trip. Product description:
```

## Product names
- Invent **fictional** brand and product names. Do not use real brands, trademarks, or real product names.
- Every product name must be unique across the whole dataset.

## Diversity requirements
Spread prompts evenly across these categories (roughly equal counts):
consumer electronics; kitchen appliances; home furniture and decor; clothing and footwear; beauty and personal care; health and fitness equipment; outdoor and camping gear; toys and games; baby and kids products; pet supplies; office supplies and stationery; tools and hardware; automotive accessories; groceries and specialty food; beverages; books and educational products; software and apps; musical instruments; jewelry and watches; garden and lawn care; travel accessories; smart home devices; arts and crafts supplies; sports equipment.

Also vary:
- **Instruction wording:** rotate between phrasings such as "Write a product description for…", "Create marketing copy for…", "Describe this product for an online store listing…", "Write a short e-commerce description of…", "Draft persuasive product copy for…". Don't use the same opening more than ~10% of the time.
- **Specifications:** 3 to 8 items, separated by semicolons. Use realistic units and numbers (dimensions, weight, capacity, materials, battery life, power, compatibility, ingredients, warranty, price where natural). Mix technical and plain-language specs.
- **Target audiences:** vary age (children's parents, teens, students, adults, seniors), expertise (beginners to professionals), budget (budget-conscious to luxury), motivation (gift buyers, eco-conscious shoppers, small business owners, hobbyists, people with specific needs such as accessibility or allergies). Make each audience a specific phrase, not just "everyone".
- **Price tier:** mix budget, mid-range, and premium products.
- **Length:** keep most prompts between 40 and 90 words.

## Don'ts
- Don't write the product descriptions themselves; only the prompts.
- Don't include harmful, regulated, or adult products (weapons, drugs, tobacco, alcohol, medical treatment claims).
- Don't add any text before or after the code block.
