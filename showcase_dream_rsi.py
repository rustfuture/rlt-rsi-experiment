from rlt_rsi.replay_simulator import ReplaySimulator, depth_first_policy, breadth_first_policy, greedy_reward_policy
import json

def main():
    print("--- Dream-RSI: History as a Replay Simulator ---")
    print("Loading historical discovery trace into the Replay Simulator...")
    sim = ReplaySimulator.from_json_trace("results/dream_trace_sample.json")
    
    print("\n[Simulating Policy 1] Depth-First Exploration")
    res_df = sim.simulate_policy(depth_first_policy, max_steps=10)
    print(f"Result: {res_df}")
    
    print("\n[Simulating Policy 2] Breadth-First Exploration")
    res_bf = sim.simulate_policy(breadth_first_policy, max_steps=10)
    print(f"Result: {res_bf}")
    
    print("\n[Simulating Policy 3] Greedy Reward-Based Exploration")
    res_greedy = sim.simulate_policy(greedy_reward_policy, max_steps=10)
    print(f"Result: {res_greedy}")
    
    print("\nNotice how different policies yield different exploration efficiencies (steps taken) to reach the maximum reward, all computed offline in milliseconds without LLM calls!")

if __name__ == "__main__":
    main()
