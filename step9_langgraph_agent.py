from typing import TypedDict, Optional

class AgentState(TypedDict):
    vlm_result: dict
    pil_image: object
    selected_defect: Optional[str]
    selected_backend: Optional[str]
    enhanced_image: Optional[object]


from PIL import Image
BACKEND_ROUTING = {
    "Blur":"deblurgan_v2",
    "Noise":"restormer",
    "Compression":"real_esrgan",
    "Lighting":"histogram_eq"
}

SEVERITY_RANK = {"None": 0, "Low": 1, "Medium": 2, "High": 3}

def find_worst_defect(vlm_result):
    worst_category = None
    worst_rank = 0

    for category in ["Blur","Noise","Compression","Lighting"]:
        info = vlm_result[category]
        if info["present"]:
            rank = SEVERITY_RANK[info["severity"]]
            if rank>worst_rank:
                worst_rank = rank
                worst_category = category
    return worst_category

def run_backend(backend_name,pil_image):
    if backend_name == "real_esrgan":
        from step8_enhancement import load_enhancer, enhance_frame
        enhancer = load_enhancer()
        return enhance_frame(enhancer,pil_image)
    print(f"'{backend_name}' is not implemented yet. Skipping.")
    return None

def node_select_backend(state: AgentState) -> AgentState:
    worst = find_worst_defect(state["vlm_result"])
    backend = BACKEND_ROUTING[worst] if worst else None
    return {**state, "selected_defect": worst, "selected_backend": backend}


def node_execute_action(state: AgentState) -> AgentState:
    if state["selected_backend"] is None:
        return state
    result = run_backend(state["selected_backend"], state["pil_image"])
    return {**state, "enhanced_image": result}

from langgraph.graph import StateGraph, END


def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("select_backend", node_select_backend)
    graph.add_node("execute_action", node_execute_action)
    graph.set_entry_point("select_backend")
    graph.add_edge("select_backend", "execute_action")
    graph.add_edge("execute_action", END)
    return graph.compile()

if __name__ == "__main__":
    from PIL import Image

    fake_vlm_result = {
        "Blur": {"present": True, "severity": "Medium", "reason": "..."},
        "Noise": {"present": False, "severity": "None", "reason": "..."},
        "Compression": {"present": True, "severity": "High", "reason": "..."},
        "Lighting": {"present": False, "severity": "None", "reason": "..."},
    }

    test_image = Image.open("test_frame.jpg")

    agent = build_agent_graph()
    initial_state = {
        "vlm_result": fake_vlm_result,
        "pil_image": test_image,
        "selected_defect": None,
        "selected_backend": None,
        "enhanced_image": None,
    }

    final_state = agent.invoke(initial_state)

    print("Selected defect:", final_state["selected_defect"])
    print("Selected backend:", final_state["selected_backend"])

    if final_state["enhanced_image"]:
        final_state["enhanced_image"].save("agent_test_output2.jpg")
        print("Saved agent_test_output2.jpg")