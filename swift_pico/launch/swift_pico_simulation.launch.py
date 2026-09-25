import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Name of the POSIX shared-memory segment carrying raw frames from the bridge
# to whycode. Both ends are given this same string; it is not a ROS topic and
# does not appear in `ros2 topic list`. Inspect it with `ls -l /dev/shm/`.
IMAGE_SHM_NAME = 'swift_pico_image_raw'


def _launch_setup(context, *args, **kwargs):

    # How raw frames reach whycode. See the image_sink comment in
    # config/mujoco_bridge.cpp for what each mode costs.
    #
    #   shm  -- shared-memory ring (default). The 11MB frame never enters DDS,
    #           so there is no serialize/deserialize pair, no Fast-DDS segment
    #           to size, and no RELIABLE-QoS backpressure able to stall the
    #           render thread when whycode falls behind.
    #   dds  -- the original sensor_msgs/Image topic. Keep for A/B measurement,
    #           and for anything outside this launch file that expects to
    #           subscribe to /image_raw.
    #   both -- ring plus topic, at the cost of one extra 11MB memcpy per
    #           frame. Bring-up and comparison only.
    image_sink = LaunchConfiguration('image_sink').perform(context)
    profile = LaunchConfiguration('profile').perform(context)
    whycode_hz = LaunchConfiguration('whycode_hz').perform(context)

    if image_sink not in ('shm', 'dds', 'both'):
        raise RuntimeError(
            f"image_sink must be one of shm|dds|both (got '{image_sink}')")

    # whycode reads the ring only when the bridge is actually writing one.
    # Empty keeps its original image_transport subscription, so the 'dds' mode
    # is byte-for-byte the pre-existing behaviour on both ends.
    whycode_shm_name = IMAGE_SHM_NAME if image_sink in ('shm', 'both') else ''

    # /image_raw is ~4.7MB/frame (1280x1280 rgb8) at up to 40Hz. Fast-DDS's
    # default shared-memory segment (~512KB) is smaller than that, so it
    # falls back to UDPv4 -- and the stock kernel UDP receive buffer is far
    # too small for that traffic, causing dropped/retransmitted fragments
    # and bursty delivery. fastdds_profile.xml raises the SHM segment above
    # the image size so the bridge and whycode_node talk over shared memory
    # instead, with no client-side kernel/sysctl changes required. Every
    # process that needs to see this large-message traffic must load it.
    fastdds_profile_path = os.path.join(
        get_package_share_directory('swift_pico'), 'config', 'fastdds_profile.xml')

    # RMW_FASTRTPS_PUBLICATION_MODE=ASYNCHRONOUS: with RELIABLE QoS, Fast-DDS's
    # default synchronous publish mode blocks the calling thread until the
    # message is handed off/ACKed -- ASYNCHRONOUS offloads that to a
    # background thread so publish() doesn't stall the render loop.
    #
    # Note: there is no MUJOCO_GL env var here -- mj_viewer.cpp always uses
    # GLFW/GLX for both the interactive window and the offscreen render, and
    # GLFW doesn't read MUJOCO_GL at all (that var only matters to MuJoCo's
    # own Python/dm_control-style headless renderers). A prior 'render_device'
    # gpu/cpu launch arg set it anyway as leftover from the deleted Python
    # bridge; it had no effect and has been removed.
    # ── MuJoCo simulation + ROS2 bridge ──────────────────────────────────────
    # A launch_ros Node execs the resolved binary directly. Deliberately NOT
    # `ExecuteProcess(cmd=['ros2', 'run', 'swift_pico', 'mujoco_bridge'], ...)`
    # -- nesting the `ros2 run` Python wrapper as a launch-managed child
    # reliably segfaults mujoco_bridge within its first couple of
    # render-thread frames once other sibling processes give the scheduler
    # less contention to mask it with (reproduced with this exact pattern in
    # swift_pico_arena_only.launch.py). Launching the executable directly
    # (no wrapper) is solid across repeated runs.
    mujoco_bridge = Node(
        package='swift_pico',
        executable='mujoco_bridge',
        output='screen',
        additional_env={
            'RMW_FASTRTPS_PUBLICATION_MODE': 'ASYNCHRONOUS',
            'FASTRTPS_DEFAULT_PROFILES_FILE': fastdds_profile_path,
        },
        parameters=[{
            'image_sink': image_sink,
            'image_shm_name': IMAGE_SHM_NAME,
            'whycode_hz': float(whycode_hz),
            # profile:=true prints a per-stage ms breakdown of the render loop
            # once a second, ending in the fps CEILING that pipeline can
            # sustain. Compare that number against whycode_hz: a target above
            # the ceiling cannot be met however the pacer is written.
            'profile': profile.lower() in ('true', '1'),
        }],
    )

    # whycode_node both subscribes to /image_raw and publishes its own
    # similarly large annotated /whycode_node/image_out, so it needs the
    # same transport profile and async publish mode as the bridge.
    whycode_env = {
        'FASTRTPS_DEFAULT_PROFILES_FILE': fastdds_profile_path,
        'RMW_FASTRTPS_PUBLICATION_MODE': 'ASYNCHRONOUS',
    }

    # ── WhyCode marker detection — legacy topics kept for controllers ────────
    whycode = Node(
        package='whycode',
        name='whycode_node',
        executable='whycode_node',
        output='screen',
        additional_env=whycode_env,
        parameters=[{
            'img_base_topic': '/image_raw',
            'info_topic': '/camera_info',
            'img_transport': 'raw',
            # WhyCon's single metric scale factor: it multiplies the reported
            # x, y AND z linearly (CTransformation::calcEigen, c2 term), so a
            # wrong value mis-scales the whole coordinate frame rather than
            # just the depth. The previous 0.2275 read ~0.85x true distance,
            # which meant a desired_state z of 12 actually parked the drone
            # under 1 m instead of 3 m, and a reported y of -2 was really
            # about -2.4 m.
            #
            # 0.2443, measured against ground truth on this configuration
            # (1920x1920, marker mesh scale 0.5, shadow pass off in arena.xml).
            # Two independent altitudes agree to 0.38%, so the scale factor is
            # effectively distance-independent here:
            #
            #     drone resting  true 14.8511 m, reported 16.5212 -> 0.24387
            #     hover z=1.428  true 13.5980 m, reported 15.0700 -> 0.24480
            #
            # It lands almost exactly on the marker's geometric black-ring
            # diameter (0.244 m): with the shadow pass disabled the binarisation
            # threshold falls essentially on the true ring edge, so the blob no
            # longer over-reads the way it did when 0.2713 was calibrated.
            #
            # WhyCon's depth goes as circle_diameter * fx / apparent_px and it
            # scales reported x, y AND z together, so re-measure whenever the
            # marker mesh scale, the lighting/shadow setup, or the binarisation
            # behaviour changes. A pure CAM_W change cancels out (fx and
            # apparent_px move together) and does NOT need a new value.
            #
            # Calibrated for the SIMULATED marker; the real hardware marker
            # needs its own calibration (see whycode_bringup.launch.py).
            # Empty unless image_sink puts frames in the ring; see above.
            'img_shm_name': whycode_shm_name,
            'img_shm_frame_id': 'camera_optical',
            'circle_diameter': 0.2443,
            'id_bits': 3,
            'id_samples': 720,
            'hamming_dist': 1,
            'num_markers': 1,
            'use_gui': True,
            'min_size': 5,
            'calib_file': '',
            'coords_method': 0,
        }],
        remappings=[
            ('~/markers', '/whycode_node/markers'),
            ('~/processed_image', '/whycode_node/image_out'),
        ],
    )



    roll_pitch_yawrate_thrust_controller = Node(
        package='rotors_control',
        namespace='rotors',
        executable='roll_pitch_yawrate_thrust_controller_node',
        name='roll_pitch_yawrate_thrust_controller',
    )

    swift_interface = Node(
        package='rotors_swift_interface',
        namespace='rotors',
        executable='rotors_swift_interface',
        name='rotors_swift_interface'
    )



    # ── Image view ───────────────────────────────────────────────────────────
    # Subscribes to whycode's large annotated image, so it needs the same
    # transport profile to receive it over shared memory too.
    #
    # width/height cap the OpenCV window size (image_view passes them to
    # cv::resizeWindow); they default to -1, which means "match the image".
    # That default was fine at 800x800, but the camera renders 1280x1280
    # (CAM_W/CAM_H in config/mujoco_bridge.cpp) -- still taller than the usable
    # height of a 1080p screen once titlebars and panels are accounted for, so
    # the window could open with its lower edge out of reach.
    #
    # `autosize` must stay False (its default) for these to take effect:
    # cv::resizeWindow is ignored on a WINDOW_AUTOSIZE window.
    #
    # Note the side effect on whycode's overlay text: CRawImage::drawStats
    # hardcodes font_scale 0.5 in SOURCE-image pixels, so it is ~11 px tall
    # whatever the resolution. Showing a 1280 frame in a 1000 window scales that
    # to ~9 px. whycode exposes no parameter for it. Raising width/height here
    # is the only knob that makes it bigger without editing that package.
    image_view = Node(
        package='image_view',
        executable='image_view',
        namespace='whycode_display',
        name='image_view',
        output='screen',
        additional_env={'FASTRTPS_DEFAULT_PROFILES_FILE': fastdds_profile_path},
        parameters=[{
            'width': 1000,
            'height': 1000,
        }],
        remappings=[('image', '/whycode_node/image_out')],
    )

    return [
        mujoco_bridge,
        whycode,
        roll_pitch_yawrate_thrust_controller,
        swift_interface,
        image_view,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'image_sink',
            default_value='shm',
            description="How raw frames reach whycode: 'shm' (shared-memory "
                        "ring, no DDS in the image path), 'dds' (the original "
                        "/image_raw topic), or 'both' (ring + topic, one extra "
                        "11MB memcpy per frame -- bring-up only).",
        ),
        DeclareLaunchArgument(
            'whycode_hz',
            default_value='30.0',
            description='Target render/publish rate for camera frames. Must '
                        'stay at or below the fps ceiling reported by '
                        'profile:=true, or the pacer simply runs late every '
                        'frame and the delivered rate settles below target.',
        ),
        DeclareLaunchArgument(
            'profile',
            default_value='false',
            description='Print a per-stage ms breakdown of the render loop '
                        'once a second, with the fps ceiling it implies.',
        ),
        OpaqueFunction(function=_launch_setup),
    ])