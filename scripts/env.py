import sys
import os
import math

import numpy as np
from isaacgym import gymapi, gymtorch, gymutil
import torch


def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)





class InitState:
    def __init__(self,
            root_pos=gymapi.Vec3(0, 0, 1),
            root_rot=gymapi.Quat(0, 0, 0, 1),
            dof_pos=np.zeros(24),
            root_vel=gymapi.Vec3(0, 0, 0),
            root_ang_vel=gymapi.Vec3(0, 0, 0),
            dof_vel=np.zeros(24)
            ):
        self.root_pos = root_pos
        self.root_rot = root_rot
        self.dof_pos = dof_pos
        self.root_vel = root_vel
        self.root_ang_vel = root_ang_vel
        self.dof_vel = dof_vel


class GenericAsset:
    def __init__(self, config):
        
        # name of actor used in the simulation
        self.name = "actor"

        self.rootpath = "assets/"
        self.filename = "urdf/robot.urdf"
        self.path = os.path.join(self.rootpath, self.filename)

        self.options = gymapi.AssetOptions()


        # Angular velocity damping for rigid bodies. Default is 0.5.
        self.options.angular_damping = 0.
        
        # The value added to the diagonal elements of inertia tensors for all of the asset’s rigid
        # bodies/links. Could improve simulation stability. Default is 0.0.
        self.options.armature = 0.
        
        # Merge links that are connected by fixed joints.
        # Specific fixed joints can be kept by adding " <... dont_collapse="true">
        self.options.collapse_fixed_joints = True
        
        # Whether to treat submeshes in the mesh as the convex decomposition of the mesh. 
        # Default is False.
        self.options.convex_decomposition_from_submeshes = False

        # Default mode used to actuate Asset joints. See isaacgym.gymapi.DriveModeFlags.
        # (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self.options.default_dof_drive_mode = 3

        # Default density parameter used for calculating mass and inertia tensor when no mass and
        # inertia data are provided, in kg/m^3. Default is 1000.0.
        self.options.density = .001

        # Disables gravity for asset. DEfault is False.
        self.options.disable_gravity = False

        # Enable gyroscopic forces. Default is True.
        self.options.enable_gyroscopic_forces = False

        # Set Asset base to a fixed placement upon import.
        # self.options.fix_base_link = False.
        self.options.fix_base_link = True

        # Switch Meshes from Z-up left-handed system to Y-up Right-handed coordinate system.
        self.options.flip_visual_attachments = False

        # Linear velocity damping for rigid bodies. Default is 0.0.
        self.options.linear_damping = 0.

        # Maximum angular velocity for rigid bodies. In rad/s. Default is 64.0.
        self.options.max_angular_velocity = 1000.

        # Maximum linear velocity for rigid bodies. In m/s. Default is 1000.0.
        self.options.max_linear_velocity = 1000.

        # How to load normals for the meshes in the asset. One of FROM_ASSET, 
        # COMPUTE_PER_VERTEX, or COMPUTE_PER_FACE. Defaults to FROM_ASSET, falls back to 
        # COMPUTE_PER_VERTEX if normals not fully specified in mesh.
        self.options.mesh_normal_mode = gymapi.MeshNormalMode.FROM_ASSET

        # Minimum mass for particles in soft bodies, in Kg. Default is 9.999999960041972e-13
        self.options.min_particle_mass = 1e-12

        # Whether to compute the center of mass from geometry and override values given in the 
        # original asset.
        self.options.override_com = False

        # Whether to compute the inertia tensor from geometry and override values given in the
        # original asset.
        self.options.override_inertia = False

        # flag to replace Cylinders with capsules for additional performance.
        self.options.replace_cylinder_with_capsule = False

        # Number of faces on generated cylinder mesh, excluding top and bottom. Default is 20.
        self.options.slices_per_cylinder = 20

        # Default tendon limit stiffness. Choose small as the limits are not implicitly solved. 
        # Avoid oscillations by setting an apporpriate damping value. Default is 1.0.
        self.options.tendon_limit_stiffness = 1.0

        # Thickness of the collision shapes. Sets how far objects should come to rest from the 
        # surface of this body. Default is 0.019999999552965164.
        self.options.thickness = 0.01

        # Whether to use materials loaded from mesh files instead of the materials defined in
        # asset file. Default False
        self.options.use_mesh_materials = True

        # Use joint space armature instead of links inertia tensor modififcations. Default is True.
        self.options.use_physx_armature = True
        
        # Whether convex decomposition is enabled. Used only with PhysX. Default False.
        self.options.vhacd_enabled = False

        # Convex decomposition parameters. Used only with PhysX. If not specified, all triangle
        # meshes will be approximated using a single convex hull.
        self.options.vhacd_params = self.options.vhacd_params


        self.init_state = InitState()





class SimulationEnv:
    def __init__(self, config, compute_device="cuda:0", graphics_device=None):

        # configs
        # TODO: move to config file
        self.num_envs = 1
        self.num_obs = 0
        self.num_privileged_obs = 0
        self.num_actions = 24

        self.dt = 1.0 / 60.0

        self.headless = False
        
        self.enable_viewer_sync = True

        self.physics_engine: gymapi.SimType = gymapi.SIM_PHYSX
        self.up_axis: gymapi.UpAxis = gymapi.UpAxis.UP_AXIS_Z

        self.use_gpu_pipeline = False
        



        self.compute_device = compute_device
        self.graphics_device = graphics_device
        if self.graphics_device is None:
            self.graphics_device = self.compute_device


        self.compute_device_type, self.compute_device_id = gymutil.parse_device_str(self.compute_device)
        self.graphics_device_type, self.graphics_device_id = gymutil.parse_device_str(self.graphics_device)

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if self.compute_device_type == "cuda" and self.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = "cpu"
        
        # graphics device for rendering, -1 for no rendering
        if self.headless == True:
            self.graphics_device_id = -1


        # initialize gym
        self.gym = gymapi.acquire_gym()

        # initialize simulation physics and graphics contexts
        self.sim = self.createSim()

        # add ground plane
        self.createGroundPlane()


        # environment handles
        self.envs = []
        self.actors = []

        self.gym.prepare_sim(self.sim)

        
        # viewer interface
        if self.headless:
            self.viewer = None
        else:
            self.viewer = self.createViewer()
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
        

    def createViewer(self):
        # create viewer
        viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if viewer is None:
            print("***[FATAL] Failed to create viewer")
            quit()

        # position the camera
        cam_pos = gymapi.Vec3(0, 1, 1)
        cam_target = gymapi.Vec3(0, 0, 0.8)
        self.gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

        return viewer
    
    def createSim(self):
        """
        Set up the simulation with configured parameters.

        Returns:
            gymapi.Sim: Simulation object
        """
        # configure sim
        sim_params = gymapi.SimParams()

        # Simulation step size
        sim_params.dt = self.dt

        # 3-Dimension vector representing gravity force in Newtons.
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)   # default Z-up

        if self.physics_engine == gymapi.SIM_FLEX:
            sim_params.flex = sim_params.flex
        elif self.physics_engine == gymapi.SIM_PHYSX:
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 6
            sim_params.physx.num_velocity_iterations = 0
            # self.sim_params.physx.num_threads = args.num_threads # default 0
            # self.sim_params.physx.use_gpu = args.use_gpu # default  False

        sim_params.up_axis = self.up_axis

        sim_params.use_gpu_pipeline = self.use_gpu_pipeline

        sim = self.gym.create_sim(self.compute_device_id, self.graphics_device_id, self.physics_engine, sim_params)

        if sim is None:
            print("***[FATAL] Failed to create sim")
            quit()

        return sim

    def createGroundPlane(self):
        """
        Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        # plane_params.static_friction = self.cfg.terrain.static_friction
        # plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        # plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)


    def loadAsset(self, asset: GenericAsset):
        print("Loading asset \"%s\"..." % (asset.path))

        self.asset_handle = self.gym.load_asset(self.sim, asset.rootpath, asset.filename, asset.options)


        self.num_bodies = self.gym.get_asset_rigid_body_count(self.asset_handle)
        self.num_dofs = self.gym.get_asset_dof_count(self.asset_handle)
        dof_props = self.gym.get_asset_dof_properties(self.asset_handle)
        rigid_shape_props = self.gym.get_asset_rigid_shape_properties(self.asset_handle)

        body_names = self.gym.get_asset_rigid_body_names(self.asset_handle)
        self.dof_names = self.gym.get_asset_dof_names(self.asset_handle)

        assert self.num_bodies == len(body_names), "Body name count mismatch"
        assert self.num_dofs == len(self.dof_names), "DOF name count mismatch"

        # contact_names = [s for s in body_names if asset.contact_name in s]
        # penalize_contact_names = []
        # for name in self.cfg.asset.penalize_contacts_on:
        #     penalized_contact_names.extend([s for s in body_names if name in s])
        # termination_contact_names = []
        # for name in self.cfg.asset.terminate_after_contacts_on:
        #     termination_contact_names.extend([s for s in body_names if name in s])


        # set up the env grid
        self.env_spacing = 2.5


        self._setEnvOrigins()




        unbounded_env = False

        if unbounded_env:
            env_lower = gymapi.Vec3(0., 0., 0.)
            env_upper = gymapi.Vec3(0., 0., 0.)
        else:
            env_lower = gymapi.Vec3(-self.env_spacing, -self.env_spacing, 0.0)
            env_upper = gymapi.Vec3(self.env_spacing, self.env_spacing, self.env_spacing)



        print("Creating %d environments" % self.num_envs)
        for i in range(self.num_envs):
            num_per_row = int(np.sqrt(self.num_envs))

            # create env
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env_handle)

            # add actor
            self.pose = gymapi.Transform()
            self.pose.p = asset.init_state.root_pos
            self.pose.r = asset.init_state.root_rot

            # 1 to disable, 0 to enable...bitwise filter
            self_collisions = 1
            actor_handle = self.gym.create_actor(env_handle, self.asset_handle, self.pose, asset.name, i, self_collisions, 0)
            self.actors.append(actor_handle)

            # set default DOF positions
            dof_state = np.zeros(self.num_dofs, dtype=gymapi.DofState.dtype)
            dof_state["pos"] = asset.init_state.dof_pos
            dof_state["vel"] = asset.init_state.dof_vel

            self.gym.set_actor_dof_states(env_handle, actor_handle, dof_state, gymapi.STATE_ALL)

        self.root_tensor = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))


    def _setEnvOrigins(self):
        """ 
        Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
        Otherwise create a grid.
        """
        pass



    def step(self, actions: torch.Tensor):
        actions = actions.astype(gymapi.DofState.dtype)
        for i in range(self.num_envs):
            # self.gym.set_dof_position_target_tensor_indexed(self.envs[i], )
            self.gym.set_actor_dof_states(self.envs[i], self.actors[i], actions, gymapi.STATE_POS)
            self.gym.set_rigid_transform(self.envs[i], 0, self.pose)

        self.render()


    def _reset_dofs(self, env_ids):
        """ 
        Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def _reset_root_states(self, env_ids):
        """
        Resets ROOT states position and velocities of selected environmments
        Sets base position based on the curriculum
        Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]

        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    
    def render(self, sync_frame_time=True):
        if not self.viewer:
            return
        
        # check for window closed
        if self.gym.query_viewer_has_closed(self.viewer):
            sys.exit()
        
        # check for keyboard events
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "QUIT" and evt.value > 0:
                sys.exit()
            elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                self.enable_viewer_sync = not self.enable_viewer_sync

        # fetch results
        if self.compute_device != "cpu":
            self.gym.fetch_results(self.sim, True)
        
        if self.enable_viewer_sync:
            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

            if sync_frame_time:
                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)
        else:
            self.gym.poll_viewer_events(self.viewer)


    def stop(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


    def run(self):


        # get array of DOF properties
        dof_states = np.zeros(self.num_dofs, dtype=gymapi.DofState.dtype)

        dof_props = self.gym.get_asset_dof_properties(self.asset_handle)

        # get list of DOF types
        dof_types = [self.gym.get_asset_dof_type(self.asset_handle, i) for i in range(self.num_dofs)]


        # get the limit-related slices of the DOF properties array
        stiffnesses = dof_props['stiffness']
        dampings = dof_props['damping']
        armatures = dof_props['armature']
        has_limits = dof_props['hasLimits']
        self.lower_limits = dof_props['lower']
        self.upper_limits = dof_props['upper']

        # initialize default positions, limits, and speeds (make sure they are in reasonable ranges)
        self.default_dof_pos = np.zeros(self.num_dofs)
        self.speeds = np.zeros(self.num_dofs)
        speed_scale = 1

        for i in range(self.num_dofs):
            if has_limits[i]:
                if dof_types[i] == gymapi.DOF_ROTATION:
                    self.lower_limits[i] = clamp(self.lower_limits[i], -math.pi, math.pi)
                    self.upper_limits[i] = clamp(self.upper_limits[i], -math.pi, math.pi)
                # make sure our default position is in range
                if self.lower_limits[i] > 0.0:
                    self.default_dof_pos[i] = self.lower_limits[i]
                elif self.upper_limits[i] < 0.0:
                    self.default_dof_pos[i] = self.upper_limits[i]
            else:
                # set reasonable animation limits for unlimited joints
                if dof_types[i] == gymapi.DOF_ROTATION:
                    # unlimited revolute joint
                    self.lower_limits[i] = -math.pi
                    self.upper_limits[i] = math.pi
                elif dof_types[i] == gymapi.DOF_TRANSLATION:
                    # unlimited prismatic joint
                    self.lower_limits[i] = -1.0
                    self.upper_limits[i] = 1.0
            # set DOF position to default
            dof_states["pos"][i] = self.default_dof_pos[i]
            # set speed depending on DOF type and range of motion
            if dof_types[i] == gymapi.DOF_ROTATION:
                self.speeds[i] = speed_scale * clamp(2 * (self.upper_limits[i] - self.lower_limits[i]), 0.25 * math.pi, 3.0 * math.pi)
            else:
                self.speeds[i] = speed_scale * clamp(2 * (self.upper_limits[i] - self.lower_limits[i]), 0.1, 7.0)

        # Print DOF properties
        for i in range(self.num_dofs):
            print("DOF %d" % i)
            print("  Name:     '%s'" % self.dof_names[i])
            print("  Type:     %s" % self.gym.get_dof_type_string(dof_types[i]))
            print("  Stiffness:  %r" % stiffnesses[i])
            print("  Damping:  %r" % dampings[i])
            print("  Armature:  %r" % armatures[i])
            print("  Limited?  %r" % has_limits[i])
            if has_limits[i]:
                print("    Lower   %f" % self.lower_limits[i])
                print("    Upper   %f" % self.upper_limits[i])





        # joint animation states
        ANIM_SEEK_LOWER = 1
        ANIM_SEEK_UPPER = 2
        ANIM_SEEK_DEFAULT = 3
        ANIM_FINISHED = 4

        # initialize animation state
        anim_state = ANIM_SEEK_LOWER
        current_dof = 0
        print("Animating DOF %d ('%s')" % (current_dof, self.dof_names[current_dof]))

        # ref_motion = np.load(ref_motion, allow_pickle=True).tolist()
        # ref_dof_pos = ref_motion["dof_pos"]
        ref_motion = None
        
        dof_positions = np.zeros(self.num_dofs)
        
        show_axis = False

        while True:
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            if not ref_motion:

                speed = self.speeds[current_dof]

                # animate the dofs
                if anim_state == ANIM_SEEK_LOWER:
                    dof_positions[current_dof] -= speed * self.dt
                    if dof_positions[current_dof] <= self.lower_limits[current_dof]:
                        dof_positions[current_dof] = self.lower_limits[current_dof]
                        anim_state = ANIM_SEEK_UPPER
                elif anim_state == ANIM_SEEK_UPPER:
                    dof_positions[current_dof] += speed * self.dt
                    if dof_positions[current_dof] >= self.upper_limits[current_dof]:
                        dof_positions[current_dof] = self.upper_limits[current_dof]
                        anim_state = ANIM_SEEK_DEFAULT
                if anim_state == ANIM_SEEK_DEFAULT:
                    dof_positions[current_dof] -= speed * self.dt
                    if dof_positions[current_dof] <= self.default_dof_pos[current_dof]:
                        dof_positions[current_dof] = self.default_dof_pos[current_dof]
                        anim_state = ANIM_FINISHED
                elif anim_state == ANIM_FINISHED:
                    dof_positions[current_dof] = self.default_dof_pos[current_dof]
                    current_dof = (current_dof + 1) % self.num_dofs
                    anim_state = ANIM_SEEK_LOWER
                    print("Animating DOF %d ('%s')" % (current_dof, self.dof_names[current_dof]))


            else:
                for i in range(24):
                    dof_positions[i] = ref_dof_pos[frame_idx, i]

                self.pose.p = gymapi.Vec3(ref_motion["root_pos"][frame_idx, 0], ref_motion["root_pos"][frame_idx, 1], ref_motion["root_pos"][frame_idx, 2])
                self.pose.r = gymapi.Quat(
                    ref_motion["root_rot"][frame_idx, 1], 
                    ref_motion["root_rot"][frame_idx, 2], 
                    ref_motion["root_rot"][frame_idx, 3],
                    ref_motion["root_rot"][frame_idx, 0]
                )


                # flip direction in Isaac Gym
                dof_positions *= -1

                frame_idx += 1
                if frame_idx >= ref_motion['root_pos'].shape[0]:
                    frame_idx = 0

            if show_axis:
                self.gym.clear_lines(self.viewer)

            for i in range(self.num_envs):
                if show_axis:
                    # get the DOF frame (origin and axis)
                    dof_handle = self.gym.get_actor_dof_handle(self.envs[i], self.actors[i], current_dof)
                    frame = self.gym.get_dof_frame(self.envs[i], dof_handle)

                    # draw a line from DOF origin along the DOF axis
                    p1 = frame.origin
                    p2 = frame.origin + frame.axis * 0.7
                    color = gymapi.Vec3(1.0, 0.0, 0.0)
                    gymutil.draw_line(p1, p2, color, self.gym, self.viewer, self.envs[i])

            self.step(dof_positions)

        print("Done")
        self.stop()
