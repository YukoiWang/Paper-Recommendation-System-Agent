import os
import json
from google import genai
from typing import List, Literal, Optional, Dict, Any
from dataclasses import dataclass, field

# ==========================================
# 1. API CONFIGURATION
# ==========================================
GOOGLE_API_KEY = (
    "AIzaSyB1-AwnpDAQEh-2CWO1XfMzQuU3v53D5ps"  # Replace with your actual key
)


# ==========================================
# 2. DATA CLASSES (THE BLACKBOARD)
# ==========================================
@dataclass
class Paper:
    title: str
    reason: str
    score: float


@dataclass
class UserProfile:
    summary: str
    interests: List[str]
    dislikes: List[str]


@dataclass
class AgentState:
    """The Shared Blackboard (bb) for all agents."""

    # --- Current State ---
    current_state: Literal["daily_push", "modify", "on_call", "idle", "none"] = "none"

    # --- User Input & Intent ---
    user_input: str = ""
    search_query: str = ""  # Populated during "on_call" demands

    # --- Recommendation Info ---
    ranked_papers: List[Paper] = field(default_factory=list)
    user_profile: UserProfile = field(default_factory=lambda: UserProfile("", [], []))

    # --- Final Output ---
    final_response: str = ""


# ==========================================
# 3. LLM SERVICE (Provided by you)
# ==========================================
class GoogleLLMService:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.5-flash"

    def generate(self, system_instruction: str, user_content: str) -> str:
        try:
            full_contents = (
                f"SYSTEM INSTRUCTION:\n{system_instruction}\n\n"
                f"USER CONTENT:\n{user_content}"
            )
            response = self.client.models.generate_content(
                model=self.model_name, contents=full_contents
            )
            return response.text
        except Exception as e:
            print(f"--- [API Error] {e} ---")
            return "{}"


# ==========================================
# 4. MOCK AGENTS (To test the Planner)
# ==========================================
class SorterAgent:
    """Mock Sorter Agent that populates the blackboard with papers."""

    def run(self, bb: AgentState):
        print("   -> [SorterAgent] Running...")
        if bb.current_state == "on_call":
            bb.ranked_papers = [
                Paper(f"New paper about {bb.search_query}", "Matches search", 0.95)
            ]
        else:
            bb.ranked_papers = [
                Paper("Standard Daily Paper", "Matches interests", 0.88)
            ]


class UIAgent:
    """Mock UI Agent that formats the final response."""

    def run(self, bb: AgentState):
        print("   -> [UIAgent] Running...")
        paper_titles = ", ".join([p.title for p in bb.ranked_papers])
        bb.final_response = (
            f"Here is your output based on {bb.current_state}. Papers: {paper_titles}"
        )


# ==========================================
# 5. THE PLANNER AGENT
# ==========================================
class PlannerAgent:
    def __init__(self, api_key: str):
        self.llm = GoogleLLMService(api_key)
        self.sorter = SorterAgent()
        self.ui = UIAgent()

    def trigger_daily_push(self, bb: AgentState):
        """Simulates the daily timer trigger."""
        print("\n=== [Planner] Triggering: DAILY PUSH ===")
        bb.current_state = "daily_push"
        bb.user_input = ""  # Clear old input

        # Route execution
        self.sorter.run(bb)
        self.ui.run(bb)
        return bb

    def handle_user_input(self, bb: AgentState, user_input: str):
        """Analyzes NLP input and routes to modify, on_call, or idle."""
        print(f"\n=== [Planner] Analyzing Input: '{user_input}' ===")
        bb.user_input = user_input

        system_prompt = """
        You are the Planner Agent. Analyze the user's input and classify their intent.
        
        CATEGORIES:
        1. "modify": The user is giving opinions/feedback on PREVIOUS recommendations (likes/dislikes).
        2. "on_call": The user is asking for a completely NEW search or topic.
        3. "idle": The user is just saying thanks, hello, or acknowledging (no action needed).

        OUTPUT JSON FORMAT:
        {
            "intent": "modify" | "on_call" | "idle",
            "profile_updates": {
                "add_interests": ["list", "of", "new", "interests"],
                "add_dislikes": ["list", "of", "dislikes"]
            },
            "search_query": "The new topic if 'on_call', otherwise empty string"
        }
        """

        user_context = (
            f"User Profile: {bb.user_profile.summary}\n" f"User Input: {user_input}"
        )

        # Call LLM
        raw_response = self.llm.generate(system_prompt, user_context)

        try:
            # Clean JSON
            clean_json = raw_response.strip().replace("```json", "").replace("```", "")
            data = json.loads(clean_json)
            intent = data.get("intent", "idle")

            if intent == "modify":
                self._execute_modify(bb, data)
            elif intent == "on_call":
                self._execute_on_call(bb, data)
            else:
                self._execute_idle(bb)

        except json.JSONDecodeError:
            print("[Planner] Failed to parse LLM intent. Defaulting to idle.")
            self._execute_idle(bb)

        return bb

    # --- Routing Logics ---

    def _execute_modify(self, bb: AgentState, llm_data: Dict):
        print("   [Planner] Intent recognized: MODIFY")
        bb.current_state = "modify"

        # 1. Modify User Profile in BB
        updates = llm_data.get("profile_updates", {})
        for i in updates.get("add_interests", []):
            if i not in bb.user_profile.interests:
                bb.user_profile.interests.append(i)
                print(f"      + Added Interest: {i}")
        for d in updates.get("add_dislikes", []):
            if d not in bb.user_profile.dislikes:
                bb.user_profile.dislikes.append(d)
                print(f"      + Added Dislike: {d}")

        # 2. Route execution
        self.sorter.run(bb)
        self.ui.run(bb)

    def _execute_on_call(self, bb: AgentState, llm_data: Dict):
        print("   [Planner] Intent recognized: ON_CALL")
        bb.current_state = "on_call"

        # 1. Update Search Query in BB
        bb.search_query = llm_data.get("search_query", "")
        print(f"      + New Search Demand: {bb.search_query}")

        # 2. Route execution
        self.sorter.run(bb)
        self.ui.run(bb)

    def _execute_idle(self, bb: AgentState):
        print("   [Planner] Intent recognized: IDLE")
        bb.current_state = "idle"
        # 1. Call no agents.
        bb.final_response = ""
        print("   [Planner] Execution stopped. Calling no agents.")


# ==========================================
# 6. TEST RUNNER
# ==========================================
if __name__ == "__main__":
    # Ensure you set the API key!
    planner = PlannerAgent(api_key=GOOGLE_API_KEY)

    # Create the shared blackboard
    blackboard = AgentState(
        user_profile=UserProfile(
            summary="CS Student", interests=["Python"], dislikes=[]
        )
    )

    # Scenario 1: Daily Timer Goes Off
    planner.trigger_daily_push(blackboard)
    print(f"Output: {blackboard.final_response}\n")

    # Scenario 2: User gives feedback (Modify)
    planner.handle_user_input(
        blackboard, "I hate these papers. Show me things about Rust instead."
    )
    print(f"Output: {blackboard.final_response}")
    print(f"Updated Dislikes: {blackboard.user_profile.dislikes}\n")

    # Scenario 3: User demands a specific topic (On-Call)
    planner.handle_user_input(
        blackboard, "Find me a paper explaining Quantum Mechanics right now."
    )
    print(f"Output: {blackboard.final_response}\n")

    # Scenario 4: User says thanks (Idle)
    planner.handle_user_input(blackboard, "Awesome, thank you!")
    print(f"Output: {blackboard.final_response}")
