#!/usr/bin/env python3
"""
Convergence Analysis Test for NSGA-II
Measures convergence behavior and diversity over generations
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from models import VRPInstance, Customer
    from io_utils import load_instance
    from distances import distance_matrix
    from multiobjective import NSGA2_VRP_Clean
except ImportError as e:
    print(f"Import error: {e}")
    print("Using fallback implementations...")
    
    # Fallback implementations
    import json
    from dataclasses import dataclass
    from typing import List
    
    @dataclass
    class Customer:
        x: float
        y: float
        demand: int
    
    @dataclass 
    class VRPInstance:
        name: str
        depot: tuple
        customers: List[Customer]
        n_vehicles: int
        capacity: int
        
    def load_instance(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        customers = [Customer(c['x'], c['y'], c['demand']) for c in data['customers']]
        
        return VRPInstance(
            name=data['name'],
            depot=tuple(data['depot']),
            customers=customers,
            n_vehicles=data['n_vehicles'],
            capacity=data['capacity']
        )
    
    def distance_matrix(depot, customers):
        n = len(customers) + 1
        matrix = np.zeros((n, n))
        
        coords = [depot] + [(c.x, c.y) for c in customers]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    x1, y1 = coords[i]
                    x2, y2 = coords[j]
                    matrix[i][j] = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        
        return matrix
    
    from multiobjective import NSGA2_VRP_Clean


def calculate_convergence_metrics(generation_stats):
    """Calculate various convergence metrics from generation statistics"""
    metrics = {
        'hypervolume_improvement': [],
        'pareto_front_stability': [],
        'diversity_trend': [],
        'objective_improvement': [],
        'convergence_rate': None,
        'stagnation_start': None
    }
    
    if not generation_stats:
        return metrics
    
    # Extract key metrics per generation
    generations = [s['generation'] for s in generation_stats]
    pareto_sizes = [s['pareto_size'] for s in generation_stats]
    diversities = [s.get('diversity', 0) for s in generation_stats]
    best_distances = [s.get('best_distance', float('inf')) for s in generation_stats if 'best_distance' in s]
    best_balances = [s.get('best_balance', float('inf')) for s in generation_stats if 'best_balance' in s]
    
    # 1. Diversity trend
    metrics['diversity_trend'] = diversities
    
    # 2. Pareto front stability (how much pareto size changes)
    if len(pareto_sizes) > 1:
        stability = []
        for i in range(1, len(pareto_sizes)):
            change = abs(pareto_sizes[i] - pareto_sizes[i-1])
            stability.append(change)
        metrics['pareto_front_stability'] = stability
    
    # 3. Objective improvement (rate of improvement in best objectives)
    if len(best_distances) > 5:  # Need some data points
        # Calculate improvement rate over sliding windows
        window_size = 5
        improvements = []
        for i in range(window_size, len(best_distances)):
            old_dist = best_distances[i-window_size]
            new_dist = best_distances[i]
            improvement = (old_dist - new_dist) / old_dist if old_dist > 0 else 0
            improvements.append(max(0, improvement))  # Only positive improvements
        metrics['objective_improvement'] = improvements
    
    # 4. Detect convergence point (when improvement stops)
    if len(best_distances) > 10:
        # Look for when improvement becomes negligible
        improvement_threshold = 0.001  # 0.1% improvement
        stagnation_threshold = 10      # 10 generations without improvement
        
        stagnation_count = 0
        for i in range(10, len(best_distances)):
            recent_best = min(best_distances[max(0, i-5):i+1])
            older_best = min(best_distances[max(0, i-15):i-5])
            
            if older_best > 0:
                improvement = (older_best - recent_best) / older_best
                if improvement < improvement_threshold:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
                
                if stagnation_count >= stagnation_threshold and metrics['stagnation_start'] is None:
                    metrics['stagnation_start'] = i
                    break
    
    # 5. Overall convergence rate (generations to reach 90% of final improvement)
    if len(best_distances) > 20:
        initial_dist = best_distances[0]
        final_dist = best_distances[-1]
        total_improvement = initial_dist - final_dist
        
        if total_improvement > 0:
            target_improvement = 0.9 * total_improvement
            target_dist = initial_dist - target_improvement
            
            for i, dist in enumerate(best_distances):
                if dist <= target_dist:
                    metrics['convergence_rate'] = i
                    break
    
    return metrics


def plot_convergence_analysis(generation_stats, convergence_metrics, instance_name):
    """Create comprehensive convergence plots"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'NSGA-II Convergence Analysis - {instance_name}', fontsize=16)
    
    generations = [s['generation'] for s in generation_stats]
    
    # Plot 1: Objective Evolution
    best_distances = [s.get('best_distance', None) for s in generation_stats]
    best_balances = [s.get('best_balance', None) for s in generation_stats]
    
    if any(d is not None for d in best_distances):
        valid_gens = [g for g, d in zip(generations, best_distances) if d is not None]
        valid_dists = [d for d in best_distances if d is not None]
        ax1.plot(valid_gens, valid_dists, 'b-o', linewidth=2, markersize=4, label='Best Distance')
    
    if any(b is not None for b in best_balances):
        valid_gens = [g for g, b in zip(generations, best_balances) if b is not None]
        valid_bals = [b for b in best_balances if b is not None]
        ax1_twin = ax1.twinx()
        ax1_twin.plot(valid_gens, valid_bals, 'r-s', linewidth=2, markersize=4, label='Best Balance')
        ax1_twin.set_ylabel('Best Balance', color='r')
        ax1_twin.tick_params(axis='y', labelcolor='r')
    
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best Distance', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title('Objective Evolution')
    ax1.grid(True, alpha=0.3)
    
    # Mark convergence point if detected
    if convergence_metrics['stagnation_start'] is not None:
        ax1.axvline(x=convergence_metrics['stagnation_start'], color='orange', 
                   linestyle='--', linewidth=2, label='Stagnation Start')
        ax1.legend()
    
    # Plot 2: Pareto Front Size Evolution
    pareto_sizes = [s['pareto_size'] for s in generation_stats]
    ax2.plot(generations, pareto_sizes, 'g-o', linewidth=2, markersize=4)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Pareto Front Size')
    ax2.set_title('Pareto Front Size Evolution')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Diversity Evolution
    diversities = [s.get('diversity', 0) for s in generation_stats]
    if any(d > 0 for d in diversities):
        ax3.plot(generations, diversities, 'm-^', linewidth=2, markersize=4)
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Population Diversity')
        ax3.set_title('Population Diversity Over Time')
        ax3.grid(True, alpha=0.3)
        
        # Mark diversity threshold
        avg_diversity = np.mean([d for d in diversities if d > 0])
        ax3.axhline(y=avg_diversity * 0.1, color='red', linestyle=':', 
                   label='Low Diversity Threshold')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No Diversity Data Available', 
                transform=ax3.transAxes, ha='center', va='center')
        ax3.set_title('Population Diversity Over Time')
    
    # Plot 4: Convergence Metrics Summary
    ax4.axis('off')
    
    # Create text summary
    summary_text = f"📊 Convergence Analysis Summary\n\n"
    
    if convergence_metrics['convergence_rate'] is not None:
        summary_text += f"🎯 90% Convergence: Generation {convergence_metrics['convergence_rate']}\n"
    else:
        summary_text += f"🎯 90% Convergence: Not reached\n"
    
    if convergence_metrics['stagnation_start'] is not None:
        summary_text += f"⏹️  Stagnation Start: Generation {convergence_metrics['stagnation_start']}\n"
    else:
        summary_text += f"⏹️  Stagnation Start: Not detected\n"
    
    final_diversity = diversities[-1] if diversities else 0
    initial_diversity = diversities[0] if diversities else 0
    summary_text += f"🌟 Initial Diversity: {initial_diversity:.4f}\n"
    summary_text += f"🌟 Final Diversity: {final_diversity:.4f}\n"
    
    if len(convergence_metrics['objective_improvement']) > 0:
        avg_improvement = np.mean(convergence_metrics['objective_improvement'])
        summary_text += f"📈 Avg Improvement Rate: {avg_improvement:.4f}\n"
    
    pareto_stability = convergence_metrics['pareto_front_stability']
    if pareto_stability:
        avg_stability = np.mean(pareto_stability)
        summary_text += f"⚖️  Pareto Stability: {avg_stability:.2f}\n"
    
    # Convergence assessment
    summary_text += f"\n🎭 Assessment:\n"
    if convergence_metrics['stagnation_start'] and convergence_metrics['stagnation_start'] < len(generations) * 0.5:
        summary_text += "❌ Early convergence detected\n"
    elif convergence_metrics['convergence_rate'] and convergence_metrics['convergence_rate'] < len(generations) * 0.3:
        summary_text += "⚡ Fast convergence\n"
    elif final_diversity > initial_diversity * 0.5:
        summary_text += "✅ Good diversity maintenance\n"
    else:
        summary_text += "⚠️  Moderate convergence\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f'convergence_analysis_{instance_name}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Convergence plot saved as: {plot_filename}")
    
    return fig


def test_convergence_analysis(instance_name: str = "small_01"):
    """Test NSGA-II convergence behavior on a specific instance"""
    
    print(f"🔬 NSGA-II Convergence Analysis - {instance_name}")
    print("=" * 60)
    
    # Load instance
    try:
        instance_path = Path(f'data/instances/{instance_name}.json')
        instance = load_instance(instance_path)
        dist_matrix = distance_matrix(instance.depot, instance.customers)
    except Exception as e:
        print(f"❌ Failed to load instance: {e}")
        return
    
    print(f"📋 Instance: {instance.name}")
    print(f"   Customers: {len(instance.customers)}")
    print(f"   Vehicles: {instance.n_vehicles}")
    print(f"   Capacity: {instance.capacity}")
    
    # Test with enhanced parameters to observe convergence
    print(f"\n🚀 Running NSGA-II with convergence tracking...")
    
    nsga2 = NSGA2_VRP_Clean(
        instance=instance,
        distance_matrix=dist_matrix,
        pop_size=40,           # Moderate population
        generations=150,       # More generations to observe convergence
        crossover_rate=0.8,
        mutation_rate=0.2,     # Higher mutation to fight convergence
        niching_radius=0.15,
        max_pareto_size=10,
        seed=42,              # Reproducible results
        verbose=True          # Track progress
    )
    
    start_time = time.time()
    population, pareto_front, info = nsga2.run()
    runtime = time.time() - start_time
    
    print(f"\n📊 Results:")
    print(f"   Runtime: {runtime:.2f}s")
    print(f"   Final Pareto size: {len(pareto_front)}")
    print(f"   Total generations: {info['generations']}")
    
    # Analyze convergence
    print(f"\n🔍 Analyzing convergence behavior...")
    generation_stats = info['statistics']
    convergence_metrics = calculate_convergence_metrics(generation_stats)
    
    # Print convergence analysis
    print(f"\n📈 Convergence Metrics:")
    if convergence_metrics['convergence_rate'] is not None:
        print(f"   🎯 90% convergence reached at generation: {convergence_metrics['convergence_rate']}")
        convergence_percentage = (convergence_metrics['convergence_rate'] / info['generations']) * 100
        print(f"   📊 Convergence speed: {convergence_percentage:.1f}% of total generations")
    else:
        print(f"   🎯 90% convergence: Not reached")
    
    if convergence_metrics['stagnation_start'] is not None:
        print(f"   ⏹️  Stagnation detected at generation: {convergence_metrics['stagnation_start']}")
        stagnation_percentage = (convergence_metrics['stagnation_start'] / info['generations']) * 100
        print(f"   📊 Stagnation point: {stagnation_percentage:.1f}% of total generations")
    else:
        print(f"   ⏹️  No clear stagnation detected")
    
    # Diversity analysis
    diversities = [s.get('diversity', 0) for s in generation_stats]
    if diversities and any(d > 0 for d in diversities):
        initial_diversity = diversities[0]
        final_diversity = diversities[-1]
        min_diversity = min(d for d in diversities if d > 0)
        
        print(f"\n🌟 Diversity Analysis:")
        print(f"   Initial diversity: {initial_diversity:.4f}")
        print(f"   Final diversity: {final_diversity:.4f}")
        print(f"   Minimum diversity: {min_diversity:.4f}")
        print(f"   Diversity retention: {(final_diversity/initial_diversity)*100:.1f}%")
        
        if final_diversity < initial_diversity * 0.1:
            print(f"   ❌ Severe diversity loss detected!")
        elif final_diversity < initial_diversity * 0.3:
            print(f"   ⚠️  Moderate diversity loss")
        else:
            print(f"   ✅ Good diversity maintenance")
    
    # Objective improvement analysis
    if convergence_metrics['objective_improvement']:
        avg_improvement = np.mean(convergence_metrics['objective_improvement'])
        print(f"\n📈 Improvement Analysis:")
        print(f"   Average improvement rate: {avg_improvement:.4f}")
        if avg_improvement < 0.001:
            print(f"   ❌ Very slow improvement (potential early convergence)")
        elif avg_improvement < 0.01:
            print(f"   ⚠️  Slow improvement")
        else:
            print(f"   ✅ Good improvement rate")
    
    # Show evolution timeline
    print(f"\n⏰ Evolution Timeline:")
    milestones = [0, 25, 50, 75, 100, info['generations']-1]
    for gen in milestones:
        if gen < len(generation_stats):
            stat = generation_stats[gen]
            diversity = stat.get('diversity', 0)
            pareto_size = stat['pareto_size']
            best_dist = stat.get('best_distance', 'N/A')
            best_bal = stat.get('best_balance', 'N/A')
            
            if isinstance(best_dist, float):
                best_dist = f"{best_dist:.1f}"
            if isinstance(best_bal, float):
                best_bal = f"{best_bal:.2f}"
                
            print(f"   Gen {gen:3d}: {pareto_size:2d} Pareto, diversity: {diversity:.4f}, "
                  f"best: ({best_dist}, {best_bal})")
    
    # Create convergence plots
    print(f"\n📊 Creating convergence visualization...")
    plot_convergence_analysis(generation_stats, convergence_metrics, instance_name)
    
    # Final assessment
    print(f"\n🎭 Final Assessment:")
    
    early_convergence = (convergence_metrics['stagnation_start'] is not None and 
                        convergence_metrics['stagnation_start'] < info['generations'] * 0.4)
    
    good_diversity = (final_diversity > initial_diversity * 0.3 if diversities else False)
    
    if early_convergence:
        print(f"   ❌ EARLY CONVERGENCE: Algorithm converged too quickly")
        print(f"   💡 Suggestions: Increase mutation rate, add restart mechanism, larger population")
    elif not good_diversity:
        print(f"   ⚠️  DIVERSITY LOSS: Population became too similar")
        print(f"   💡 Suggestions: Enhance diversity preservation, better niching")
    else:
        print(f"   ✅ GOOD CONVERGENCE: Balanced exploration and exploitation")
    
    return {
        'instance': instance_name,
        'runtime': runtime,
        'pareto_size': len(pareto_front),
        'convergence_metrics': convergence_metrics,
        'generation_stats': generation_stats,
        'early_convergence': early_convergence,
        'diversity_maintained': good_diversity
    }


if __name__ == "__main__":
    print("🧬 NSGA-II Convergence Analysis Tool")
    print("=" * 50)
    
    # Test convergence on small dataset
    result = test_convergence_analysis("small_01")
    
    print(f"\n🏁 Analysis Complete!")
    print(f"   Results saved to convergence_analysis_small_01.png")
