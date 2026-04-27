"""
Theory of Mind (TOM) Prompt Templates

These prompts analyze posts and comments through the lens of Theory of Mind,
identifying values and mental state attributions across different TOM categories:
- Beliefs: What the author thinks is true or false
- Desires: What the author wants, prefers, or values
- Intentions: What the author plans to do or wants others to do
- Emotions: What the author feels or what emotions they attribute to others
- Knowledge: What the author knows, doesn't know, or believes others know
- Perspective-taking: How the author considers other viewpoints
"""

# ============================================================================
# TOM PROMPT TEMPLATE FOR POSTS
# ============================================================================

TOM_POST_PROMPT_TEMPLATE = """You are analyzing a post from r/ChangeMyView (CMV), a Reddit subreddit where users post opinions they hold but are open to changing. The purpose of CMV is for people to have their views challenged through thoughtful, respectful debate.

Your task is to analyze this CMV post through the lens of Theory of Mind (TOM). Theory of Mind refers to the ability to understand and attribute mental states—beliefs, desires, intentions, emotions, and knowledge—to oneself and others.

<post>
Title: {title}

Body: {body}
</post>

## TOM Categories Explained:

1. **BELIEFS**: What the author thinks is true or false. This includes:
   - Factual beliefs about the world
   - Moral beliefs about right and wrong
   - Causal beliefs about how things work
   - Beliefs about other people's mental states
   - False beliefs (understanding that others may hold incorrect beliefs)

2. **DESIRES**: What the author wants, prefers, or values. This includes:
   - Personal preferences and goals
   - What outcomes they hope for
   - What they wish were different
   - What they prioritize or care about most

3. **INTENTIONS**: What the author plans to do or wants others to do. This includes:
   - Explicit action plans
   - Implied goals or purposes
   - What they're trying to achieve
   - What they want others to do or change

4. **EMOTIONS**: What the author feels or what emotions they attribute to others. This includes:
   - Their own emotional states (frustration, hope, concern, etc.)
   - Emotions they believe others experience
   - Emotional appeals they make
   - How emotions influence their reasoning

5. **KNOWLEDGE**: What the author knows, doesn't know, or believes others know. This includes:
   - Claims about what is known or unknown
   - Epistemic confidence or uncertainty
   - What information they think others have
   - Knowledge gaps they acknowledge

6. **PERSPECTIVE-TAKING**: How the author considers other viewpoints. This includes:
   - Acknowledging alternative perspectives
   - Understanding others' reasoning
   - Empathy or consideration of others' situations
   - Recognition of complexity or multiple valid viewpoints

Instructions for analysis:

For each TOM category, identify:
- **Values**: What principles, preferences, or priorities are expressed in this category? (Summarize each value in 1-4 words, comma-separated)
- **Content**: What concrete beliefs, desires, intentions, emotions, or knowledge claims are made? (Brief description)

Output Format:

<analysis>
2-3 relatively short sentences of analysis and reasoning.
</analysis>

<beliefs>
Values: Value1, Value2, Value3 (or "none" if no clear values)
Content: Brief description of key beliefs expressed
</beliefs>

<desires>
Values: Value1, Value2, Value3 (or "none" if no clear values)
Content: Brief description of key desires or preferences expressed
</desires>

<intentions>
Values: Value1, Value2, Value3 (or "none" if no clear values)
Content: Brief description of key intentions or goals expressed
</intentions>

<emotions>
Values: Value1, Value2, Value3 (or "none" if no clear values)
Content: Brief description of emotional states or appeals expressed
</emotions>

<knowledge>
Values: Value1, Value2, Value3 (or "none" if no clear values)
Content: Brief description of knowledge claims or epistemic stances expressed
</knowledge>

<perspective_taking>
Values: Value1, Value2, Value3 (or "none" if no clear values)
Content: Brief description of how other perspectives are considered
</perspective_taking>

IMPORTANT: 
- Output all six TOM category tags (beliefs, desires, intentions, emotions, knowledge, perspective_taking)
- For each category, provide both Values (comma-separated) and Content (brief description)
- Use "none" if a category has no clear content
- Do NOT put explanations within the value tags, only the final values
"""

# Part 1: analysis + beliefs, desires, intentions (shorter output to reduce truncation)
TOM_POST_PROMPT_PART1 = """You are analyzing a post from r/ChangeMyView (CMV) through Theory of Mind (TOM).

<post>
Title: {title}

Body: {body}
</post>

Analyze ONLY these three TOM categories (keep output brief):

1. **BELIEFS**: What the author thinks is true/false. Values (1-4 words each, comma-separated) and Content (brief).
2. **DESIRES**: What the author wants/values. Values and Content.
3. **INTENTIONS**: What the author plans or wants others to do. Values and Content.

Output Format:

<analysis>
2-3 relatively short sentences of analysis and reasoning.
</analysis>

<beliefs>
Values: Value1, Value2 (or "none")
Content: Brief description
</beliefs>

<desires>
Values: Value1, Value2 (or "none")
Content: Brief description
</desires>

<intentions>
Values: Value1, Value2 (or "none")
Content: Brief description
</intentions>

IMPORTANT: Output only these four tags. Be concise.
"""

# Part 2: emotions, knowledge, perspective_taking
TOM_POST_PROMPT_PART2 = """You are analyzing a post from r/ChangeMyView (CMV) through Theory of Mind (TOM).

<post>
Title: {title}

Body: {body}
</post>

Analyze ONLY these three TOM categories (keep output brief):

4. **EMOTIONS**: What the author feels or attributes to others. Values and Content.
5. **KNOWLEDGE**: What the author knows/doesn't know. Values and Content.
6. **PERSPECTIVE-TAKING**: How the author considers other viewpoints. Values and Content.

Output Format:

<emotions>
Values: Value1, Value2 (or "none")
Content: Brief description
</emotions>

<knowledge>
Values: Value1, Value2 (or "none")
Content: Brief description
</knowledge>

<perspective_taking>
Values: Value1, Value2 (or "none")
Content: Brief description
</perspective_taking>

IMPORTANT: Output only these three tags. Be concise.
"""


# ============================================================================
# TOM PROMPT TEMPLATE FOR COMMENTS
# ============================================================================

TOM_COMMENT_PROMPT_TEMPLATE = """You are analyzing a comment from r/ChangeMyView (CMV), a Reddit subreddit where users post opinions they hold but are open to changing. In CMV, commenters try to present arguments that might change the original poster's mind.

Your task is to analyze this comment through the lens of Theory of Mind (TOM). Theory of Mind refers to the ability to understand and attribute mental states—beliefs, desires, intentions, emotions, and knowledge—to oneself and others.

<context>
Original Post Title: {title}
</context>

<comment>
{body}
</comment>

## TOM Categories Explained:

1. **BELIEFS**: What the author thinks is true or false. This includes:
   - Factual beliefs about the world
   - Moral beliefs about right and wrong
   - Causal beliefs about how things work
   - Beliefs about other people's mental states
   - False beliefs (understanding that others may hold incorrect beliefs)

2. **DESIRES**: What the author wants, prefers, or values. This includes:
   - Personal preferences and goals
   - What outcomes they hope for
   - What they wish were different
   - What they prioritize or care about most

3. **INTENTIONS**: What the author plans to do or wants others to do. This includes:
   - Explicit action plans
   - Implied goals or purposes
   - What they're trying to achieve
   - What they want others to do or change

4. **EMOTIONS**: What the author feels or what emotions they attribute to others. This includes:
   - Their own emotional states (frustration, hope, concern, etc.)
   - Emotions they believe others experience
   - Emotional appeals they make
   - How emotions influence their reasoning

5. **KNOWLEDGE**: What the author knows, doesn't know, or believes others know. This includes:
   - Claims about what is known or unknown
   - Epistemic confidence or uncertainty
   - What information they think others have
   - Knowledge gaps they acknowledge

6. **PERSPECTIVE-TAKING**: How the author considers other viewpoints. This includes:
   - Acknowledging alternative perspectives
   - Understanding others' reasoning
   - Empathy or consideration of others' situations
   - Recognition of complexity or multiple valid viewpoints

Instructions for analysis:

For each TOM category, identify:
- **Values**: What principles, preferences, or priorities are expressed in this category? (Summarize each value in 1-4 words, comma-separated)
- **Content**: What concrete beliefs, desires, intentions, emotions, or knowledge claims are made? (Brief description)

Output Format:

<analysis>
2-3 relatively short sentences of analysis and reasoning.
</analysis>

<beliefs>
Values: Value1, Value2, Value3 (or "none" if no clear values)
Content: Brief description of key beliefs expressed
</beliefs>

<desires>
Values: Value1, Value2, Value3 (or "none" if no clear values)
Content: Brief description of key desires or preferences expressed
</desires>

<intentions>
Values: Value1, Value2, Value3 (or "none" if no clear values)
Content: Brief description of key intentions or goals expressed
</intentions>

<emotions>
Values: Value1, Value2, Value3 (or "none" if no clear values)
Content: Brief description of emotional states or appeals expressed
</emotions>

<knowledge>
Values: Value1, Value2, Value3 (or "none" if no clear values)
Content: Brief description of knowledge claims or epistemic stances expressed
</knowledge>

<perspective_taking>
Values: Value1, Value2, Value3 (or "none" if no clear values)
Content: Brief description of how other perspectives are considered
</perspective_taking>

IMPORTANT: 
- Output all six TOM category tags (beliefs, desires, intentions, emotions, knowledge, perspective_taking)
- For each category, provide both Values (comma-separated) and Content (brief description)
- Use "none" if a category has no clear content
- Do NOT put explanations within the value tags, only the final values
"""

# Part 1: beliefs, desires, intentions
TOM_COMMENT_PROMPT_PART1 = """You are analyzing a comment from r/ChangeMyView (CMV) through Theory of Mind (TOM).

<context>
Original Post Title: {title}
</context>

<comment>
{body}
</comment>

Analyze ONLY these three TOM categories (keep output brief):

1. **BELIEFS**: What the author thinks is true/false. Values and Content.
2. **DESIRES**: What the author wants/values. Values and Content.
3. **INTENTIONS**: What the author plans or wants others to do. Values and Content.

Output Format:

<analysis>
2-3 relatively short sentences of analysis and reasoning.
</analysis>

<beliefs>
Values: Value1, Value2 (or "none")
Content: Brief description
</beliefs>

<desires>
Values: Value1, Value2 (or "none")
Content: Brief description
</desires>

<intentions>
Values: Value1, Value2 (or "none")
Content: Brief description
</intentions>

IMPORTANT: Output only these four tags. Be concise.
"""

# Part 2: emotions, knowledge, perspective_taking
TOM_COMMENT_PROMPT_PART2 = """You are analyzing a comment from r/ChangeMyView (CMV) through Theory of Mind (TOM).

<context>
Original Post Title: {title}
</context>

<comment>
{body}
</comment>

Analyze ONLY these three TOM categories (keep output brief):

4. **EMOTIONS**: What the author feels or attributes to others. Values and Content.
5. **KNOWLEDGE**: What the author knows/doesn't know. Values and Content.
6. **PERSPECTIVE-TAKING**: How the author considers other viewpoints. Values and Content.

Output Format:

<emotions>
Values: Value1, Value2 (or "none")
Content: Brief description
</emotions>

<knowledge>
Values: Value1, Value2 (or "none")
Content: Brief description
</knowledge>

<perspective_taking>
Values: Value1, Value2 (or "none")
Content: Brief description
</perspective_taking>

IMPORTANT: Output only these three tags. Be concise.
"""


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_tom_post_prompt(post: dict, template: str = TOM_POST_PROMPT_TEMPLATE) -> str:
    """Build the TOM prompt for analyzing a CMV post."""
    title = post.get("title", "")
    body = post.get("selftext", "")
    return template.format(title=title, body=body)


def build_tom_comment_prompt(
    post_title: str,
    comment: dict,
    template: str = TOM_COMMENT_PROMPT_TEMPLATE,
) -> str:
    """Build the TOM prompt for analyzing a single comment."""
    body = (comment.get("body") or "")[:4000]  # Truncate long comments
    return template.format(title=post_title, body=body)


def merge_parsed_tom(part1: dict, part2: dict) -> dict:
    """Merge two parsed TOM responses (part1: analysis+beliefs+desires+intentions; part2: emotions+knowledge+perspective_taking) into one."""
    default_cat = {"values": None, "content": None}
    merged = {
        "analysis": part1.get("analysis"),
        "beliefs": part1.get("beliefs", default_cat),
        "desires": part1.get("desires", default_cat),
        "intentions": part1.get("intentions", default_cat),
        "emotions": part2.get("emotions", default_cat),
        "knowledge": part2.get("knowledge", default_cat),
        "perspective_taking": part2.get("perspective_taking", default_cat),
    }
    truncated = part1.get("truncated") or part2.get("truncated")
    if truncated:
        merged["truncated"] = True
        cats = list(part1.get("truncated_categories") or []) + list(part2.get("truncated_categories") or [])
        if cats:
            merged["truncated_categories"] = cats
    return merged


def parse_tom_response(response: str) -> dict:
    """
    Parse the TOM response into structured format.
    
    Returns:
        dict with keys: analysis, beliefs, desires, intentions, emotions, knowledge, perspective_taking
        Each category contains: values (str), content (str)
        Also includes: truncated (bool) flag if response appears incomplete
    """
    result = {
        "analysis": None,
        "beliefs": {"values": None, "content": None},
        "desires": {"values": None, "content": None},
        "intentions": {"values": None, "content": None},
        "emotions": {"values": None, "content": None},
        "knowledge": {"values": None, "content": None},
        "perspective_taking": {"values": None, "content": None},
        "raw_response": response,
        "truncated": False,
    }
    
    # Check for truncation indicators
    categories = ["beliefs", "desires", "intentions", "emotions", "knowledge", "perspective_taking"]
    required_tags = ["<analysis>", "</analysis>"] + [f"<{cat}>" for cat in categories] + [f"</{cat}>" for cat in categories]
    
    # Check if all closing tags are present
    missing_closing_tags = []
    for category in categories:
        tag = f"<{category}>"
        end_tag = f"</{category}>"
        if tag in response and end_tag not in response:
            missing_closing_tags.append(category)
    
    # Check if analysis tag is closed
    if "<analysis>" in response and "</analysis>" not in response:
        missing_closing_tags.append("analysis")
    
    # Check if response ends abruptly (common truncation pattern)
    response_ends_abruptly = (
        response.rstrip().endswith("</") or
        response.rstrip().endswith("<") or
        (len(response) > 100 and response[-50:].count("<") > response[-50:].count(">"))
    )
    
    if missing_closing_tags or response_ends_abruptly:
        result["truncated"] = True
        if missing_closing_tags:
            result["truncated_categories"] = missing_closing_tags
    
    # Extract analysis
    if "<analysis>" in response and "</analysis>" in response:
        start = response.find("<analysis>") + len("<analysis>")
        end = response.find("</analysis>")
        result["analysis"] = response[start:end].strip()
    elif "<analysis>" in response:
        # Incomplete analysis tag
        start = response.find("<analysis>") + len("<analysis>")
        result["analysis"] = response[start:].strip()
        result["truncated"] = True
    
    # Extract each TOM category
    for category in categories:
        tag = f"<{category}>"
        end_tag = f"</{category}>"
        
        if tag in response:
            start = response.find(tag) + len(tag)
            if end_tag in response:
                end = response.find(end_tag)
                content = response[start:end].strip()
                
                # Parse Values and Content
                if "Values:" in content:
                    values_part = content.split("Values:")[1]
                    if "Content:" in values_part:
                        values_str = values_part.split("Content:")[0].strip()
                        content_str = values_part.split("Content:")[1].strip()
                    else:
                        values_str = values_part.strip()
                        content_str = None
                    
                    result[category]["values"] = values_str if values_str and values_str.lower() != "none" else None
                    result[category]["content"] = content_str if content_str else None
                else:
                    # Fallback: treat entire content as description
                    result[category]["content"] = content if content.lower() != "none" else None
            else:
                # Incomplete tag - extract what we can
                content = response[start:].strip()
                if "Values:" in content:
                    values_part = content.split("Values:")[1]
                    if "Content:" in values_part:
                        values_str = values_part.split("Content:")[0].strip()
                        content_str = values_part.split("Content:")[1].strip()
                    else:
                        values_str = values_part.strip()
                        content_str = None
                    
                    result[category]["values"] = values_str if values_str and values_str.lower() != "none" else None
                    result[category]["content"] = content_str if content_str else None
                else:
                    result[category]["content"] = content if content.lower() != "none" else None
                result["truncated"] = True
    
    return result
