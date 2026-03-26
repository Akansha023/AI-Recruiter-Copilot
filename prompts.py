"""
Prompt templates for AI Recruiter Copilot.
Each function returns a crafted prompt string for Gemini.
"""


def summary_prompt(context: str) -> str:
    return f"""
You are a senior technical recruiter with 10+ years of experience evaluating candidates.

RESUME CONTEXT:
{context}

Task: Summarize this candidate in exactly 5 bullet points.
Focus on:
- Total years of experience
- Core technical skills
- Most impressive achievement or project
- Industry/domain background
- Career trajectory / growth

Format each bullet with an emoji prefix. Be concise, specific, and factual.
"""


def strengths_risks_prompt(context: str, jd: str) -> str:
    return f"""
You are a senior technical recruiter evaluating candidate fit.

RESUME CONTEXT:
{context}

JOB DESCRIPTION:
{jd}

Task: Provide a structured strengths and risks analysis.

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:

**TOP 3 STRENGTHS:**
1. [Strength with specific evidence from resume]
2. [Strength with specific evidence from resume]
3. [Strength with specific evidence from resume]

**TOP 3 RISKS / GAPS:**
1. [Risk with explanation of impact on job performance]
2. [Risk with explanation of impact on job performance]
3. [Risk with explanation of impact on job performance]

**OVERALL ASSESSMENT:**
[2-3 sentence summary of whether this candidate is worth moving forward]
"""


def questions_prompt(context: str, jd: str) -> str:
    return f"""
You are an expert technical interviewer preparing for a candidate interview.

RESUME CONTEXT:
{context}

JOB DESCRIPTION:
{jd}

Task: Generate 5 targeted interview questions based on this specific candidate's background.

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:

**TECHNICAL QUESTIONS:**
1. [Specific technical question tied to their experience]
   → What to look for: [Key answer signals]

2. [Another technical question]
   → What to look for: [Key answer signals]

**BEHAVIORAL QUESTIONS:**
3. [Behavioral question using STAR method]
   → What to look for: [Key answer signals]

4. [Another behavioral question]
   → What to look for: [Key answer signals]

**DEEP-DIVE QUESTION:**
5. [A challenging, scenario-based question that tests depth]
   → What to look for: [Key answer signals]
"""


def scoring_prompt(context: str, jd: str) -> str:
    return f"""
You are a data-driven technical recruiter scoring a candidate objectively.

RESUME CONTEXT:
{context}

JOB DESCRIPTION:
{jd}

Task: Score this candidate across 4 dimensions. Be objective and strict — not everyone deserves top marks.

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:

**SCORING BREAKDOWN:**

| Dimension         | Score | Reasoning |
|-------------------|-------|-----------|
| Skill Match       |  /5   | [Reason]  |
| Experience Level  |  /5   | [Reason]  |
| Job Stability     |  /5   | [Reason]  |
| Domain Relevance  |  /5   | [Reason]  |

**FINAL SCORE: [X] / 20**

**RECOMMENDATION:**
[ ] Strong Hire   [ ] Hire   [ ] Maybe   [ ] No Hire

**REASONING:**
[2-3 sentences justifying your recommendation]
"""


def compare_prompt(context1: str, context2: str, jd: str) -> str:
    return f"""
You are a senior recruiter comparing two candidates for the same role.

CANDIDATE A RESUME:
{context1}

CANDIDATE B RESUME:
{context2}

JOB DESCRIPTION:
{jd}

Task: Compare both candidates head-to-head and recommend who to move forward.

FORMAT YOUR RESPONSE LIKE THIS:

**HEAD-TO-HEAD COMPARISON:**
| Criteria          | Candidate A | Candidate B |
|-------------------|-------------|-------------|
| Technical Skills  | [rating]    | [rating]    |
| Experience        | [rating]    | [rating]    |
| Cultural Fit      | [rating]    | [rating]    |
| Growth Potential  | [rating]    | [rating]    |

**WINNER: [Candidate A / Candidate B / Tie]**

**REASONING:**
[3-4 sentences explaining the decision]
"""
