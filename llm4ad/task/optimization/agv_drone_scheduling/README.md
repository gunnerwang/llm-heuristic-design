# AGV and Drone Scheduling System

This module implements a mixed-fleet scheduling system for AGVs and drones in a manufacturing environment. The goal is to efficiently coordinate the movement of parts and trays between different locations while managing battery levels and handling special scenarios like traffic congestion and processing failures.

## Environment Overview

The environment consists of:

- **3 Locations**: A (loading area), B (processing area), and C (buffer area)
- **Mixed Fleet**: 4 AGVs and 2 drones with different capabilities
- **Transport Tasks**: Moving parts and trays between locations
- **Battery Management**: Vehicles need to monitor and recharge their batteries
- **Special Features**: Traffic congestion, processing failures, and specialized vehicle roles

## Key Files

- `multi_vehicle_env.py` - The environment simulation with all vehicle dynamics
- `evaluation.py` - Evaluation framework for scheduling algorithms
- `template.py` - Template for implementing scheduling algorithms
- `multi_vehicle_test_train_result.py` - Testing script for individual schedulers
- `test_agv_drone_scheduling.py` - Comprehensive test comparing different schedulers

## Scheduling Challenges

The scheduling system must handle:

1. **Battery Management**: Vehicles must charge before they run out of power
2. **Traffic Congestion**: AGVs are more affected by traffic than drones
3. **Processing Failures**: Some parts may experience processing failures, requiring special handling
4. **Vehicle Specialization**: Efficiently use the different capabilities of AGVs and drones
5. **Location-Based Decision Making**: Making smart choices based on current vehicle locations

## Available Actions

- Action 0: Move part and tray from A to B (for processing)
- Action 1: Move processed part and tray from B to A (recycle tray)
- Action 2: Move part and tray from A to C (buffer storage)
- Action 3: Move part and tray from C to B (from buffer to processing)
- Action 4: Go to charging station (at location A) for recharging

## Running Tests

To compare different scheduling strategies:

```bash
python test_agv_drone_scheduling.py --instances 10 --verbose
```

Options:
- `--instances N`: Number of test instances for each scheduler (default: 5)
- `--verbose`: Show detailed output during simulation
- `--seed N`: Set random seed for reproducibility

## Implementing New Schedulers

To implement a new scheduling algorithm, follow the template in `template.py`. Your scheduler must implement the function:

```python
def select_next_action(env, vehicle_index, current_node):
    # Your scheduling logic here
    # Return action index (0-4) or -1 if no valid action
```

## Performance Metrics

Schedulers are evaluated based on:
- Completion time (lower is better)
- Total parts processed
- Battery efficiency
- Charging frequency
- Ability to handle processing failures 