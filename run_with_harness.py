# Run with harness

from agent_harness import(
    AgentHarness,
    print_harness_report
)
def create_initial_state()->dict:
    return {
        "user_request": (),
        "number_of_people": 4,
        "booking_time": "Tomorrow at 8:00 PM",
        "selected_rastraunt":None,
        "attepmted_rastront":[],
        "plan":"",
        "tool_result": None,
        "goal_achieved": False,
        "evaluation_message":"",
        "final_answer":"",
        "iteration":0,
        "max_iterations":5,

    }



def main() -> None:
    harness=AgentHarness(
        graph=rastront_agent,
        agent_name="rastront-booking-agent-v1",
        audit_log_path=("audit_logs/rastront_agent.jsonl")
        
        # Operational Limits
        max_iterations=5
        max_tool_calls=5,
        
        # Max expected LLM calls
        max_llm_calls=5
        max_total_tokens=10_000,
        
        # Total Graph execution timeout
        timeout_seconds=90,

        # Same meaningfull state cannot appear more than twice
        repeated_state_limit=2,

        # Same rastront should not be selected twice
        repeated_action_limit=1,

        # Langgraph step level protection
        recursion_limit=30,
    )

    result=harness.run(
        initial_state=create_initial_state(),
        metadata={
            "environment": "development",
            "application": "rastront-demo",
            "agent_version": "1.0.0",
            "model_provider": "azure-openai",
        },

        raise_on_error=False
    )

    print_harness_report(result)
    if not result.succeeded:
        raise SystemExit(1)

if __name__ =="__main__":
    main()