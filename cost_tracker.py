"""
Cost Tracker - Track your API costs in real-time during development
Use this in all your projects to stay on budget!
"""

import json
from datetime import datetime
from pathlib import Path


class CostTracker:
    """Track API costs across your development session"""
    
    # Pricing per 1M tokens (update these if pricing changes)
    PRICING = {
        'claude-haiku': {'input': 0.80, 'output': 4.00},
        'claude-sonnet': {'input': 3.00, 'output': 15.00},
        'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
        'gpt-4o': {'input': 2.50, 'output': 10.00},
        'embedding-small': {'input': 0.02, 'output': 0.00},
    }
    
    def __init__(self, log_file='cost_log.json'):
        self.log_file = Path(log_file)
        self.session_costs = []
        self._load_history()
    
    def _load_history(self):
        """Load previous cost history"""
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = []
    
    def _save_history(self):
        """Save cost history to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def track_call(self, model, input_tokens, output_tokens, description=""):
        """
        Track a single API call
        
        Args:
            model: Model name (e.g., 'claude-haiku', 'gpt-4o-mini')
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
            description: What this call was for (optional)
        """
        if model not in self.PRICING:
            print(f"⚠️  Unknown model: {model}")
            return
        
        pricing = self.PRICING[model]
        input_cost = (input_tokens / 1_000_000) * pricing['input']
        output_cost = (output_tokens / 1_000_000) * pricing['output']
        total_cost = input_cost + output_cost
        
        call_data = {
            'timestamp': datetime.now().isoformat(),
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost,
            'description': description
        }
        
        self.session_costs.append(call_data)
        self.history.append(call_data)
        self._save_history()
        
        return total_cost
    
    def print_session_summary(self):
        """Print summary of current session costs"""
        if not self.session_costs:
            print("No API calls tracked in this session")
            return
        
        total_cost = sum(call['total_cost'] for call in self.session_costs)
        total_tokens = sum(call['total_tokens'] for call in self.session_costs)
        num_calls = len(self.session_costs)
        
        print("\n" + "="*60)
        print("📊 SESSION SUMMARY")
        print("="*60)
        print(f"Total API calls: {num_calls}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Total cost: ${total_cost:.4f}")
        print(f"Average cost per call: ${total_cost/num_calls:.6f}")
        print("="*60)
        
        # Breakdown by model
        models = {}
        for call in self.session_costs:
            model = call['model']
            if model not in models:
                models[model] = {'calls': 0, 'cost': 0, 'tokens': 0}
            models[model]['calls'] += 1
            models[model]['cost'] += call['total_cost']
            models[model]['tokens'] += call['total_tokens']
        
        print("\nBreakdown by model:")
        for model, stats in models.items():
            print(f"  {model}:")
            print(f"    Calls: {stats['calls']}")
            print(f"    Tokens: {stats['tokens']:,}")
            print(f"    Cost: ${stats['cost']:.4f}")
        print("="*60 + "\n")
    
    def print_project_summary(self, project_name="Project"):
        """Print summary of all costs for this project"""
        if not self.history:
            print("No API calls tracked yet")
            return
        
        total_cost = sum(call['total_cost'] for call in self.history)
        total_tokens = sum(call['total_tokens'] for call in self.history)
        num_calls = len(self.history)
        
        print("\n" + "="*60)
        print(f"📈 {project_name.upper()} - TOTAL COSTS")
        print("="*60)
        print(f"Total API calls: {num_calls}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Total cost: ${total_cost:.2f}")
        print("="*60 + "\n")
    
    def estimate_remaining_budget(self, total_budget=25.00):
        """Estimate remaining budget from free credits"""
        spent = sum(call['total_cost'] for call in self.history)
        remaining = total_budget - spent
        
        print(f"\n💰 BUDGET STATUS")
        print(f"   Total budget: ${total_budget:.2f}")
        print(f"   Spent so far: ${spent:.2f}")
        print(f"   Remaining: ${remaining:.2f}")
        
        if remaining < 5:
            print(f"   ⚠️  Warning: Running low on budget!")
        elif remaining < 1:
            print(f"   🚨 ALERT: Less than $1 remaining!")
        else:
            print(f"   ✅ You're good!")
        
        return remaining


# Example usage
if __name__ == "__main__":
    # Create tracker
    tracker = CostTracker(log_file='project1_costs.json')
    
    # Track some example calls
    print("Simulating some API calls...\n")
    
    tracker.track_call('claude-haiku', 1000, 500, "Testing RAG retrieval")
    tracker.track_call('claude-haiku', 2000, 1000, "Testing with larger context")
    tracker.track_call('embedding-small', 5000, 0, "Embedding document chunks")
    
    # Print summaries
    tracker.print_session_summary()
    tracker.print_project_summary("Project 1: Personal Knowledge RAG")
    tracker.estimate_remaining_budget(total_budget=25.00)
    
    print("\n✅ All costs logged to: project1_costs.json")
    print("   You can review this anytime to see where your money went!\n")
