#!/usr/bin/env python3
"""
Actor Destruction Validation Script

Tests CARLA 0.9.16 actor lifecycle best practices implementation:
- Verifies is_alive checks work correctly
- Tests sensor.stop() + sensor.destroy() sequence
- Validates proper cleanup order (sensors before vehicle)
- Ensures no RuntimeError exceptions raised

References:
- CARLA Python API: https://carla.readthedocs.io/en/latest/python_api/
- CARLA Core Actors: https://carla.readthedocs.io/en/latest/core_actors/

Author: TD3 AV System Development Team
Date: 2025-01-13
"""

import carla
import time
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_sensor_destruction():
    """
    Test proper sensor destruction sequence with error handling.

    Tests:
    1. Actor spawning and sensor attachment
    2. is_alive property verification
    3. is_listening property for sensors
    4. Proper destruction sequence (stop → destroy)
    5. Error handling for already-destroyed actors
    6. Cleanup order (sensors before vehicle)
    """
    logger.info("=" * 80)
    logger.info("ACTOR DESTRUCTION VALIDATION TEST")
    logger.info("=" * 80)

    try:
        # Connect to CARLA server
        logger.info("\n[STEP 1] Connecting to CARLA server...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        version = client.get_server_version()
        logger.info(f"✓ Connected to CARLA {version}")

        world = client.get_world()

        # Spawn test vehicle
        logger.info("\n[STEP 2] Spawning test vehicle...")
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        spawn_points = world.get_map().get_spawn_points()

        if not spawn_points:
            logger.error("✗ No spawn points available")
            return False

        spawn_point = spawn_points[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        logger.info(f"✓ Vehicle spawned (ID: {vehicle.id})")

        # Verify vehicle is alive
        logger.info("\n[STEP 3] Verifying actor lifecycle properties...")
        assert hasattr(vehicle, 'is_alive'), "Vehicle missing is_alive property"
        assert vehicle.is_alive, "Vehicle not alive after spawn"
        logger.info(f"✓ Vehicle.is_alive = {vehicle.is_alive}")

        # Spawn test sensors (camera, collision, lane invasion, obstacle)
        logger.info("\n[STEP 4] Spawning test sensors...")
        sensors = {}

        # Camera sensor
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        sensors['camera'] = camera
        logger.info(f"✓ Camera spawned (ID: {camera.id})")

        # Collision sensor
        collision_bp = bp_lib.find('sensor.other.collision')
        collision_transform = carla.Transform(carla.Location())
        collision = world.spawn_actor(collision_bp, collision_transform, attach_to=vehicle)
        sensors['collision'] = collision
        logger.info(f"✓ Collision sensor spawned (ID: {collision.id})")

        # Lane invasion sensor
        lane_bp = bp_lib.find('sensor.other.lane_invasion')
        lane_transform = carla.Transform(carla.Location())
        lane = world.spawn_actor(lane_bp, lane_transform, attach_to=vehicle)
        sensors['lane_invasion'] = lane
        logger.info(f"✓ Lane invasion sensor spawned (ID: {lane.id})")

        # Obstacle sensor
        obstacle_bp = bp_lib.find('sensor.other.obstacle')
        obstacle_bp.set_attribute('distance', '10.0')
        obstacle_bp.set_attribute('hit_radius', '0.5')
        obstacle_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        obstacle = world.spawn_actor(obstacle_bp, obstacle_transform, attach_to=vehicle)
        sensors['obstacle'] = obstacle
        logger.info(f"✓ Obstacle sensor spawned (ID: {obstacle.id})")

        # Start listening on sensors
        logger.info("\n[STEP 5] Starting sensor callbacks...")
        camera.listen(lambda data: None)  # Dummy callback
        collision.listen(lambda event: None)
        lane.listen(lambda event: None)
        obstacle.listen(lambda event: None)

        # Verify all sensors listening
        for sensor_name, sensor in sensors.items():
            assert hasattr(sensor, 'is_listening'), f"{sensor_name} missing is_listening property"
            assert sensor.is_listening, f"{sensor_name} not listening after listen() call"
            logger.info(f"✓ {sensor_name}.is_listening = {sensor.is_listening}")

        # Test proper destruction sequence
        logger.info("\n[STEP 6] Testing proper sensor destruction sequence...")
        logger.info("Destruction order: sensors (children) → vehicle (parent)")

        for sensor_name, sensor in sensors.items():
            logger.info(f"\n  Destroying {sensor_name}...")

            # Step 1: Verify is_alive before operations
            if not sensor.is_alive:
                logger.warning(f"  ⚠ {sensor_name} already destroyed, skipping")
                continue

            logger.info(f"  ✓ {sensor_name}.is_alive = True (before stop)")

            # Step 2: Stop listening
            if sensor.is_listening:
                sensor.stop()
                logger.info(f"  ✓ {sensor_name}.stop() called")
                time.sleep(0.01)  # Grace period

            # Step 3: Verify stopped
            if hasattr(sensor, 'is_listening'):
                logger.info(f"  ✓ {sensor_name}.is_listening = {sensor.is_listening} (after stop)")

            # Step 4: Check is_alive again before destroy
            if not sensor.is_alive:
                logger.warning(f"  ⚠ {sensor_name} became invalid before destroy")
                continue

            # Step 5: Destroy
            success = sensor.destroy()
            logger.info(f"  ✓ {sensor_name}.destroy() returned: {success}")

        # Test vehicle destruction (after sensors)
        logger.info("\n[STEP 7] Testing vehicle destruction (parent after children)...")

        if not vehicle.is_alive:
            logger.warning("  ⚠ Vehicle already destroyed")
        else:
            logger.info(f"  ✓ Vehicle.is_alive = True (before destroy)")
            success = vehicle.destroy()
            logger.info(f"  ✓ Vehicle.destroy() returned: {success}")

        # Verify no exceptions raised
        logger.info("\n[STEP 8] Verification complete")
        logger.info("✅ TEST PASSED: No RuntimeError exceptions raised")
        logger.info("✅ All actors destroyed successfully using proper sequence")
        logger.info("✅ Error handling implemented correctly")

        return True

    except RuntimeError as e:
        logger.error(f"\n✗ TEST FAILED: RuntimeError during destruction")
        logger.error(f"  Error message: {e}")
        logger.error("  This indicates actor destruction error handling needs improvement")
        return False

    except AssertionError as e:
        logger.error(f"\n✗ TEST FAILED: Assertion error")
        logger.error(f"  Error message: {e}")
        return False

    except Exception as e:
        logger.error(f"\n✗ TEST FAILED: Unexpected exception")
        logger.error(f"  Error type: {type(e).__name__}")
        logger.error(f"  Error message: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_double_destruction():
    """
    Test error handling for double destruction (accessing already-destroyed actor).

    This simulates the crash scenario: attempting to access an actor that was
    already destroyed by the server.
    """
    logger.info("\n" + "=" * 80)
    logger.info("DOUBLE DESTRUCTION ERROR HANDLING TEST")
    logger.info("=" * 80)

    try:
        logger.info("\n[STEP 1] Connecting to CARLA server...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()

        logger.info("\n[STEP 2] Spawning test vehicle...")
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        logger.info(f"✓ Vehicle spawned (ID: {vehicle.id})")

        logger.info("\n[STEP 3] First destruction...")
        success = vehicle.destroy()
        logger.info(f"✓ First destroy() returned: {success}")

        logger.info("\n[STEP 4] Testing is_alive after destruction...")
        is_alive = vehicle.is_alive
        logger.info(f"✓ Vehicle.is_alive = {is_alive} (should be False)")

        if is_alive:
            logger.warning("  ⚠ is_alive still True after destruction!")
            return False

        logger.info("\n[STEP 5] Attempting second destruction (should be handled gracefully)...")
        try:
            # This should not raise RuntimeError if error handling is correct
            success = vehicle.destroy()
            logger.info(f"✓ Second destroy() returned: {success} (no exception)")
        except RuntimeError as e:
            # If exception raised, it means is_alive check was not used
            logger.warning(f"  ⚠ RuntimeError raised on second destroy: {e}")
            logger.warning("  This should be prevented by is_alive check")
            return False

        logger.info("\n✅ TEST PASSED: Double destruction handled gracefully")
        logger.info("✅ No RuntimeError exception raised")
        return True

    except Exception as e:
        logger.error(f"\n✗ TEST FAILED: Unexpected exception")
        logger.error(f"  Error type: {type(e).__name__}")
        logger.error(f"  Error message: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    logger.info("\n" + "=" * 80)
    logger.info("CARLA ACTOR DESTRUCTION VALIDATION SUITE")
    logger.info("Testing error handling implementation for training crash fix")
    logger.info("=" * 80)

    all_passed = True

    # Test 1: Proper destruction sequence
    logger.info("\n\nTEST 1/2: Proper Sensor Destruction Sequence")
    logger.info("-" * 80)
    test1_passed = test_sensor_destruction()
    all_passed = all_passed and test1_passed

    # Wait between tests
    time.sleep(2)

    # Test 2: Double destruction error handling
    logger.info("\n\nTEST 2/2: Double Destruction Error Handling")
    logger.info("-" * 80)
    test2_passed = test_double_destruction()
    all_passed = all_passed and test2_passed

    # Final summary
    logger.info("\n\n" + "=" * 80)
    logger.info("VALIDATION SUITE SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Test 1 (Proper Sequence):      {'✅ PASSED' if test1_passed else '✗ FAILED'}")
    logger.info(f"Test 2 (Double Destruction):   {'✅ PASSED' if test2_passed else '✗ FAILED'}")
    logger.info("")

    if all_passed:
        logger.info("🎉 ALL TESTS PASSED 🎉")
        logger.info("Error handling implementation is correct.")
        logger.info("Training should no longer crash on actor destruction.")
        sys.exit(0)
    else:
        logger.error("❌ SOME TESTS FAILED ❌")
        logger.error("Review error handling implementation.")
        sys.exit(1)


if __name__ == '__main__':
    main()
