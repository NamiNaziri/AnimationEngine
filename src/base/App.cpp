#define _CRT_SECURE_NO_WARNINGS

#include "App.hpp"
#include "base/Main.hpp"
#include "gpu/GLContext.hpp"
#include "gpu/Buffer.hpp"
#include "utility.hpp"

#include <array>
#include <cassert>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstddef>
#include <type_traits>
#include <numeric>
#include <map>
#include <Eigen/Dense>


using namespace FW;
using namespace std;

namespace {

enum VertexShaderAttributeLocations
{
	ATTRIB_POSITION = 0,
	ATTRIB_NORMAL = 1,
	ATTRIB_COLOR = 2,
	ATTRIB_JOINTS1 = 3,
	ATTRIB_JOINTS2 = 4,
	ATTRIB_WEIGHTS1 = 5,
	ATTRIB_WEIGHTS2 = 6
};

} // namespace

App::App(void)
:   common_ctrl_			(CommonControls::Feature_Default & ~CommonControls::Feature_RepaintOnF5),
	drawmode_				(MODE_SKELETON),
	filename_				(),
	shading_toggle_			(false),
	shading_mode_changed_	(false),
	KEY_X_Hold				(false),
	KEY_Y_Hold				(false),
	KEY_Z_Hold				(false),
	solve_IK_				(false),
	solve_IK_changed		(false),
	take_snapshot_			(false),
	take_snapshot_changed	(false),
	camera_rotation_		(FW_PI),
	selected_joint_			(0)
{
	static_assert(is_standard_layout<Vertex>::value, "Vertex must be standard layout to use offsetof");
	static_assert(is_standard_layout<WeightedVertex>::value, "WeightedVertex must be standard layout to use offsetof");

	static const Vec3f distinct_colors[6] = {
		Vec3f(0, 0, 1), Vec3f(0, 1, 0), Vec3f(0, 1, 1),
		Vec3f(1, 0, 0), Vec3f(1, 0, 1), Vec3f(1, 1, 0)};
	for (auto i = 0u; i < 100; ++i)
		joint_colors_.push_back(distinct_colors[i % 6]);

	String filename = window_.showFileLoadDialog("Load model");
	if (filename.getLength())
	{
		if (filename.endsWith(".bvhobj") || filename.endsWith(".bvh") || filename.endsWith(".names") || filename.endsWith(".weights"))
			loadAnimation(filename);
		else
			loadModel(filename);
	}

	initRendering();
		
	common_ctrl_.showFPS(true);
	common_ctrl_.addToggle(&take_snapshot_, FW_KEY_I, "snap shot (K)", &take_snapshot_changed);
	common_ctrl_.addToggle(&solve_IK_, FW_KEY_I, "Solve IK (I)", &solve_IK_changed);
	common_ctrl_.addSeparator();
	common_ctrl_.addToggle((S32*)&drawmode_, MODE_SKELETON,			FW_KEY_1, "Draw joints and bones (1)");
	common_ctrl_.addToggle((S32*)&drawmode_, MODE_MESH_CPU,			FW_KEY_2, "Draw mesh, SSD on CPU (2)");
	common_ctrl_.addToggle((S32*)&drawmode_, MODE_MESH_GPU,			FW_KEY_3, "EXTRA: Draw mesh, SSD on GPU (3)");
	common_ctrl_.addToggle((S32*)&drawmode_, MODE_MESH_GPU_DQ,      FW_KEY_4, "Draw mesh, DQ on GPU (4)");
	common_ctrl_.addSeparator();
	common_ctrl_.addToggle(&animationMode,							FW_KEY_A, "Animate mesh (A)", &animationMode_changed);
	common_ctrl_.addToggle(&shading_toggle_,						FW_KEY_T, "Toggle shading mode (T)",	&shading_mode_changed_);
	

	window_.setTitle("Assignment 3");

	window_.addListener(this);
	window_.addListener(&common_ctrl_);

	window_.setSize( Vec2i(800, 800) );

	Vec3f selected_joint_pos = (skel_.getToWorldTransform(selected_joint_) * Vec4f(0.f, 0.f, 0.f, 1.f)).getXYZ();
	IKTargetPointLocation = selected_joint_pos;

	timer_.start();
}

bool App::handleEvent(const Window::Event& ev) {	

	if (shading_mode_changed_) {
		common_ctrl_.message(shading_toggle_ ?
			"Directional light shading using normals; direction to light (0.5, 0.5, -0.6)" :
			"Joint weight coloring");
		shading_mode_changed_ = false;
	}

	if (solve_IK_changed)
	{
		if (solve_IK_)
		{
			skel_.SolveIK(selected_joint_, IKTargetPointLocation);
			solve_IK_ = false;
		}
		solve_IK_changed = false;
	}

	if (take_snapshot_changed)
	{
		if (take_snapshot_)
		{
			skel_.TakeKeyframeSnapShot();
			take_snapshot_ = false;
		}
		take_snapshot_changed = false;
	}

	if (animationMode_changed)
	{
		skel_.SetAnimationMode(animationMode);
		animationMode_changed = false;
	}


	if (ev.type == Window::EventType_KeyDown) {
		if (ev.key == FW_KEY_HOME)
			camera_rotation_ -= 0.05 * FW_PI;
		else if (ev.key == FW_KEY_END)
			camera_rotation_ += 0.05 * FW_PI;

		static const float rot_incr = 0.05;
		if (ev.key == FW_KEY_UP)
			skel_.incrJointRotation(selected_joint_, Vec3f(rot_incr, 0, 0));
		else if (ev.key == FW_KEY_DOWN)
			skel_.incrJointRotation(selected_joint_, Vec3f(-rot_incr, 0, 0));
		else if (ev.key == FW_KEY_LEFT)
			skel_.incrJointRotation(selected_joint_, Vec3f(0, rot_incr, 0));
		else if (ev.key == FW_KEY_RIGHT)
			skel_.incrJointRotation(selected_joint_, Vec3f(0, -rot_incr, 0));
		else if (ev.key == FW_KEY_PAGE_UP)
			skel_.incrJointRotation(selected_joint_, Vec3f(0, 0, rot_incr));
		else if (ev.key == FW_KEY_PAGE_DOWN)
			skel_.incrJointRotation(selected_joint_, Vec3f(0, 0, -rot_incr));
		else if (ev.key == FW_KEY_R)
		{
			skel_.setJointRotation(selected_joint_, Vec3f(0.0f));
			
			Vec3f selected_joint_pos = (skel_.getToWorldTransform(selected_joint_) * Vec4f(0.f, 0.f, 0.f, 1.f)).getXYZ();
			IKTargetPointLocation = selected_joint_pos;
		}
		else if (ev.key == FW_KEY_Q)
			selected_joint_ = selected_joint_ == 0 ? 0 : selected_joint_-1;
		else if (ev.key == FW_KEY_W)
			selected_joint_ = clamp(selected_joint_+1, 0u, (unsigned)skel_.getNumJoints()-1u);
		else if (ev.key == FW_KEY_MOUSE_RIGHT) {
			SelectBoneByMouse(ev);
		}
		else if (ev.key == FW_KEY_X)
		{
			clip_mouse_pos_ = Vec2f(ev.mousePos) / Vec2f(window_.getSize()) * 2.0f - 1.0f;
			clip_mouse_pos_.y *= -1;

			if (KEY_X_Hold == false)
			{
				init_mouse_to_world_pos_.x = FindMouseToWorld(clip_mouse_pos_).x;
			
			}
			KEY_X_Hold = true;
			FindPointInCameraXYplane(0);
		}
		else if (ev.key == FW_KEY_Y)
		{
			clip_mouse_pos_ = Vec2f(ev.mousePos) / Vec2f(window_.getSize()) * 2.0f - 1.0f;
			clip_mouse_pos_.y *= -1;

			if (KEY_Y_Hold == false)
			{
				init_mouse_to_world_pos_.y = FindMouseToWorld(clip_mouse_pos_).y;

			}
			KEY_Y_Hold = true;
			FindPointInCameraXYplane(1);
		}
		else if (ev.key == FW_KEY_Z)
		{
			clip_mouse_pos_ = Vec2f(ev.mousePos) / Vec2f(window_.getSize()) * 2.0f - 1.0f;
			clip_mouse_pos_.y *= -1;

			if (KEY_Z_Hold == false)
			{
				init_mouse_to_world_pos_.z = FindMouseToWorld(clip_mouse_pos_).z;

			}
			KEY_Z_Hold = true;
			FindPointInCameraXYplane(2);
		}
	}

	if (ev.type == Window::EventType_KeyUp) {
		if (ev.key == FW_KEY_X)
		{
			KEY_X_Hold = false;
		}
		else if (ev.key == FW_KEY_Y)
		{
			KEY_Y_Hold = false;
		}
		else if (ev.key == FW_KEY_Z)
		{
			KEY_Z_Hold = false;
		}
	}
	
	if (ev.type == Window::EventType_Close)	{
		window_.showModalMessage("Exiting...");
		delete this;
		return true;
	}

	if (ev.type == Window::EventType_Mouse && ev.mouseDragging)
		camera_rotation_ += 0.01f * ev.mouseDelta.x;

	window_.setVisible(true);
	if (ev.type == Window::EventType_Paint)
		render();
	window_.repaint();
	return false;
}

void App::initRendering()
{
	// Ask the Nvidia framework for the GLContext object associated with the window.
	// As a side effect, this initializes the OpenGL context and lets us call GL functions.
	auto ctx = window_.getGL();

	animationStart = GetTickCount();
	
	// Create vertex attribute objects and buffers for vertex data.
	glGenVertexArrays(1, &gl_.simple_vao);
	glGenVertexArrays(1, &gl_.ssd_vao);
	glGenVertexArrays(1, &gl_.dq_vao);
	glGenBuffers(1, &gl_.simple_vertex_buffer);
	glGenBuffers(1, &gl_.ssd_vertex_buffer);
	glGenBuffers(1, &gl_.dq_vertex_buffer);
	
	// Set up vertex attribute object for doing SSD on the CPU. The data will be loaded and re-loaded later, on each frame.
	glBindVertexArray(gl_.simple_vao);
	glBindBuffer(GL_ARRAY_BUFFER, gl_.simple_vertex_buffer);
	glEnableVertexAttribArray(ATTRIB_POSITION);
	glVertexAttribPointer(ATTRIB_POSITION, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*) 0);
	glEnableVertexAttribArray(ATTRIB_NORMAL);
	glVertexAttribPointer(ATTRIB_NORMAL, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*) offsetof(Vertex, normal));
	glEnableVertexAttribArray(ATTRIB_COLOR);
	glVertexAttribPointer(ATTRIB_COLOR, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*) offsetof(Vertex, color));
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	
	// Set up vertex attribute object for doing SSD on the GPU, and load all data to buffer.
	glBindVertexArray(gl_.ssd_vao);
	glBindBuffer(GL_ARRAY_BUFFER, gl_.ssd_vertex_buffer);
	glEnableVertexAttribArray(ATTRIB_POSITION);
	glVertexAttribPointer(ATTRIB_POSITION, 3, GL_FLOAT, GL_FALSE, sizeof(WeightedVertex), (GLvoid*) 0);
	glEnableVertexAttribArray(ATTRIB_NORMAL);
	glVertexAttribPointer(ATTRIB_NORMAL, 3, GL_FLOAT, GL_FALSE, sizeof(WeightedVertex), (GLvoid*) offsetof(WeightedVertex, normal));
	glEnableVertexAttribArray(ATTRIB_COLOR);
	glVertexAttribPointer(ATTRIB_COLOR, 3, GL_FLOAT, GL_FALSE, sizeof(WeightedVertex), (GLvoid*) offsetof(WeightedVertex, color));
	glEnableVertexAttribArray(ATTRIB_JOINTS1);
	glVertexAttribIPointer(ATTRIB_JOINTS1, 4, GL_INT, sizeof(WeightedVertex), (GLvoid*) offsetof(WeightedVertex, joints[0]));
	glEnableVertexAttribArray(ATTRIB_JOINTS2);
	glVertexAttribIPointer(ATTRIB_JOINTS2, 4, GL_INT, sizeof(WeightedVertex), (GLvoid*) offsetof(WeightedVertex, joints[4]));
	glEnableVertexAttribArray(ATTRIB_WEIGHTS1);
	glVertexAttribPointer(ATTRIB_WEIGHTS1, 4, GL_FLOAT, GL_FALSE, sizeof(WeightedVertex), (GLvoid*) offsetof(WeightedVertex, weights[0]));
	glEnableVertexAttribArray(ATTRIB_WEIGHTS2);
	glVertexAttribPointer(ATTRIB_WEIGHTS2, 4, GL_FLOAT, GL_FALSE, sizeof(WeightedVertex), (GLvoid*) offsetof(WeightedVertex, weights[4]));

	glBufferData(GL_ARRAY_BUFFER, sizeof(WeightedVertex) * weighted_vertices_.size(), weighted_vertices_.data(), GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	// Set up vertex attribute object for doing dual quaternion SSD on the GPU, and load all data to buffer.
	glBindVertexArray(gl_.dq_vao);
	glBindBuffer(GL_ARRAY_BUFFER, gl_.dq_vertex_buffer);
	glEnableVertexAttribArray(ATTRIB_POSITION);
	glVertexAttribPointer(ATTRIB_POSITION, 3, GL_FLOAT, GL_FALSE, sizeof(WeightedVertex), (GLvoid*)0);
	glEnableVertexAttribArray(ATTRIB_NORMAL);
	glVertexAttribPointer(ATTRIB_NORMAL, 3, GL_FLOAT, GL_FALSE, sizeof(WeightedVertex), (GLvoid*)offsetof(WeightedVertex, normal));
	glEnableVertexAttribArray(ATTRIB_COLOR);
	glVertexAttribPointer(ATTRIB_COLOR, 3, GL_FLOAT, GL_FALSE, sizeof(WeightedVertex), (GLvoid*)offsetof(WeightedVertex, color));
	glEnableVertexAttribArray(ATTRIB_JOINTS1);
	glVertexAttribIPointer(ATTRIB_JOINTS1, 4, GL_INT, sizeof(WeightedVertex), (GLvoid*)offsetof(WeightedVertex, joints[0]));
	glEnableVertexAttribArray(ATTRIB_JOINTS2);
	glVertexAttribIPointer(ATTRIB_JOINTS2, 4, GL_INT, sizeof(WeightedVertex), (GLvoid*)offsetof(WeightedVertex, joints[4]));
	glEnableVertexAttribArray(ATTRIB_WEIGHTS1);
	glVertexAttribPointer(ATTRIB_WEIGHTS1, 4, GL_FLOAT, GL_FALSE, sizeof(WeightedVertex), (GLvoid*)offsetof(WeightedVertex, weights[0]));
	glEnableVertexAttribArray(ATTRIB_WEIGHTS2);
	glVertexAttribPointer(ATTRIB_WEIGHTS2, 4, GL_FLOAT, GL_FALSE, sizeof(WeightedVertex), (GLvoid*)offsetof(WeightedVertex, weights[4]));

	glBufferData(GL_ARRAY_BUFFER, sizeof(WeightedVertex) * weighted_vertices_.size(), weighted_vertices_.data(), GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);


	// Compile and link the shader programs.

	auto simple_prog = new GLContext::Program(
		"#version 330\n"
		FW_GL_SHADER_SOURCE(
		layout(location = 0) in vec4 aPosition;
		layout(location = 1) in vec3 aNormal;
		layout(location = 2) in vec3 aColor;

		out vec4 vColor;

		uniform mat4 uWorldToClip;
		uniform float uShadingMix;

		const vec3 directionToLight = normalize(vec3(0.5, 0.5, 0.6));

		void main()
		{
			float clampedCosine = clamp(dot(aNormal, directionToLight), 0.0, 1.0);
			vec3 litColor = vec3(clampedCosine);
			vColor = vec4(mix(aColor.xyz, litColor, uShadingMix), 1);
			gl_Position = uWorldToClip * aPosition;
		}
		),
		"#version 330\n"
		FW_GL_SHADER_SOURCE(
		in vec4 vColor;
		out vec4 fColor;
		void main()
		{
			fColor = vColor;
		}
		));
	ctx->setProgram("simple_shader", simple_prog);
	// YOUR CODE HERE (EXTRA):
	// Perform the skinning math in the vertex shader, like the example binary
	// does. This is the exact same thing you already did in R4 & R5, only
	// translated from C++ to GLSL. Remember to handle normals as well.
	auto ssd_prog = new GLContext::Program(
		"#version 330\n"
		FW_GL_SHADER_SOURCE(
		layout(location = 0) in vec4 aPosition;
		layout(location = 1) in vec3 aNormal;
		layout(location = 2) in vec4 aColor;
		layout(location = 3) in ivec4 aJoints1;
		layout(location = 4) in ivec4 aJoints2;
		layout(location = 5) in vec4 aWeights1;
		layout(location = 6) in vec4 aWeights2;

		const vec3 directionToLight = normalize(vec3(0.5, 0.5, 0.6));
		uniform mat4 uWorldToClip;
		uniform float uShadingMix;

		out vec4 vColor;		

		const int numJoints = 100;
		const int WEIGHTS_PER_VERTEX = 8;
		uniform mat4 uJoints[numJoints];
	
		void main()
		{
			
			
			vec4 SkinnedPosition = vec4(0.f);
			vec3 SkinnedNormal = vec3(0.f);
			vec4 SkinnedColor = vec4(0.f);
			for (int i = 0; i < 8; i++)
			{
				if (i < 4)
				{
					if (aWeights1[i] != 0)
					{
						SkinnedPosition += aWeights1[i] * uJoints[aJoints1[i]] * aPosition;
						SkinnedNormal += (aWeights1[i] * uJoints[aJoints1[i]] * vec4(aNormal, 0.f)).xyz;
						SkinnedColor += (aWeights1[i] * aColor);
					}
				}
				else
				{
					if (aWeights2[i%4] != 0)
					{
						SkinnedPosition += aWeights2[i % 4] * uJoints[aJoints2[i % 4]] * aPosition;
						SkinnedNormal += (aWeights2[i % 4] * uJoints[aJoints2[i % 4]] * vec4(aNormal, 0.f)).xyz;
						SkinnedColor += (aWeights2[i % 4] * aColor);
					}
				}
				
			}


			float clampedCosine = clamp(dot(SkinnedNormal, directionToLight), 0.0, 1.0);
			vec3 litColor = vec3(clampedCosine);
			vColor = vec4(mix(SkinnedColor.xyz, litColor, uShadingMix), 1);
			gl_Position = uWorldToClip * SkinnedPosition;

		}
		),
		"#version 330\n"
		FW_GL_SHADER_SOURCE(
		in vec4 vColor;
		out vec4 fColor;
		void main()
		{
			fColor = vColor;
		}
		));
	ctx->setProgram("ssd_shader", ssd_prog);

	// Get the IDs of the shader programs and their uniform input locations from OpenGL.
	gl_.ssd_shader = ssd_prog->getHandle();
	gl_.ssd_transforms_uniform = glGetUniformLocation(gl_.ssd_shader, "uJoints");
	gl_.ssd_world_to_clip_uniform = glGetUniformLocation(gl_.ssd_shader, "uWorldToClip");
	gl_.ssd_shading_mix_uniform = glGetUniformLocation(gl_.ssd_shader, "uShadingMix");
	gl_.simple_shader = simple_prog->getHandle();
	gl_.simple_world_to_clip_uniform = glGetUniformLocation(gl_.simple_shader, "uWorldToClip");
	gl_.simple_shading_mix_uniform = glGetUniformLocation(gl_.simple_shader, "uShadingMix");


	/////////////////////////////////////////



	auto dq_prog = new GLContext::Program(
		"#version 330\n"
		FW_GL_SHADER_SOURCE(
	layout(location = 0) in vec4 aPosition;
	layout(location = 1) in vec3 aNormal;
	layout(location = 2) in vec4 aColor;
	layout(location = 3) in ivec4 aJoints1;
	layout(location = 4) in ivec4 aJoints2;
	layout(location = 5) in vec4 aWeights1;
	layout(location = 6) in vec4 aWeights2;

	const vec3 directionToLight = normalize(vec3(0.5, 0.5, 0.6));
	uniform mat4 uWorldToClip;
	uniform float uShadingMix;

	out vec4 vColor;

	const int numJoints = 100;
	const int WEIGHTS_PER_VERTEX = 8;
	uniform vec4 uDualQuatertions_0[numJoints];
	uniform vec4 uDualQuatertions_e[numJoints];

	void main()
	{
		vec4 b0 = vec4(0.f);
		vec4 be = vec4(0.f);
		for (int i = 0; i < 8; i++)
		{
			if (i < 4)
			{
				
				b0 += aWeights1[i] * uDualQuatertions_0[aJoints1[i]];
				be += aWeights1[i] * uDualQuatertions_e[aJoints1[i]];
			}
			else
			{
				b0 += aWeights2[i % 4] * uDualQuatertions_0[aJoints2[i % 4]];
				be += aWeights2[i % 4] * uDualQuatertions_e[aJoints2[i % 4]];
			}

		}
		
		vec4 c0 = b0 / length(b0);
		vec4 ce = be / length(b0);

		float x0 = c0[0];
		float y0 = c0[1];
		float z0 = c0[2];
		float w0 = c0[3];


		float xe = ce[0];
		float ye = ce[1];
		float ze = ce[2];
		float we = ce[3];

		float t0 = 2. * (-1. * we * x0 + xe * w0 - ye * z0 + ze * y0);
		float t1 = 2. * (-1. * we * y0 + xe * z0 + ye * w0 - ze * x0);
		float t2 = 2. * (-1. * we * z0 - xe * y0 + ye * x0 + ze * w0);

		vec4 c1 = vec4(1. - 2. * y0 * y0 - 2. * z0 * z0,
			2. * x0 * y0 + 2. * w0 * z0,
			2. * x0 * z0 - 2. * w0 * y0,
			0.);

		vec4 c2 = vec4(2. * x0 * y0 - 2. * w0 * z0,
			1. - 2. * x0 * x0 - 2. * z0 * z0,
			2. * y0 * z0 + 2. * w0 * x0,
			0.);

		vec4 c3 = vec4(2. * x0 * z0 + 2. * w0 * y0,
			2. * y0 * z0 - 2. * w0 * x0,
			1. - 2. * x0 * x0 - 2. * y0 * y0,
			0.);

		vec4 c4 = vec4(t0, t1, t2, 1.);


		mat4 tranformMat = mat4(	c1,  // 1. column
							c2,  // 2. column
							c3,  // 3. column
							c4); // 4. column
		
		vec4 SkinnedPosition = tranformMat * aPosition;
		vec3 SkinnedNormal = (tranformMat * vec4(aNormal, 0.f)).xyz;
		vec4 SkinnedColor = aColor;


		float clampedCosine = clamp(dot(SkinnedNormal, directionToLight), 0.0, 1.0);
		vec3 litColor = vec3(clampedCosine);
		vColor = vec4(mix(SkinnedColor.xyz, litColor, uShadingMix), 1);
		gl_Position = uWorldToClip * SkinnedPosition;
		
	}
	),
		"#version 330\n"
		FW_GL_SHADER_SOURCE(
			in vec4 vColor;
	out vec4 fColor;
	void main()
	{
		fColor = vColor;
	}
	));
	ctx->setProgram("dq_shader", dq_prog);

	// Get the IDs of the shader programs and their uniform input locations from OpenGL.
	gl_.dq_shader = dq_prog->getHandle();
	gl_.dq_dualquaternions_0_uniform = glGetUniformLocation(gl_.dq_shader, "uDualQuatertions_0");
	gl_.dq_dualquaternions_e_uniform = glGetUniformLocation(gl_.dq_shader, "uDualQuatertions_e");
	gl_.dq_world_to_clip_uniform = glGetUniformLocation(gl_.dq_shader, "uWorldToClip");
	gl_.dq_shading_mix_uniform = glGetUniformLocation(gl_.dq_shader, "uShadingMix");



}

void App::render() {
	// Clear screen
	glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Enable depth testing
	glEnable(GL_DEPTH_TEST);
	
	
	float deltaTime = timer_.end();
	//std::cout << deltaTime << std::endl;

	if (animationMode)
	{
		int animFrame = int((GetTickCount() - animationStart) / 16.666668f);

		skel_.AddTime(deltaTime);
		skel_.setAnimationFrame(animFrame);
	}

	// Adjust viewport and aspect ratio to window
	auto sz = window_.getSize();
	float fAspect = float(sz.x)/sz.y;
	glViewport(0, 0, sz.x, sz.y);
	
	// Set up transform matrices.
	// They will be fed to the shader program via uniform variables.

	// World space -> clip space transform: simple projection and camera.

	// Our camera orbits around (0.5, 0.5, 0.5) at a fixed distance.
	static const float camera_distance = 0.8f;	
	Mat4f C;
	const Mat3f rot = Mat3f::rotation(Vec3f(0, 1, 0), -camera_rotation_);
	C.setCol(0, Vec4f(rot.getCol(0), 0));
	C.setCol(1, Vec4f(rot.getCol(1), 0));
	C.setCol(2, Vec4f(rot.getCol(2), 0));
	C.setCol(3, Vec4f(0, 0, camera_distance, 1));
	C = C * Mat4f::translate(Vec3f(-0.5f));

	Mat4f P;
	static const float fNear = 0.1f, fFar = 40.0f;
	P.setCol(0, Vec4f(1, 0, 0, 0));
	P.setCol(1, Vec4f(0, fAspect, 0, 0));
	P.setCol(2, Vec4f(0, 0, (fFar+fNear)/(fFar-fNear), 1));
	P.setCol(3, Vec4f(0, 0, -2*fFar*fNear/(fFar-fNear), 0));
	Mat4f world_to_clip = P * C;

	if (drawmode_ == MODE_SKELETON) {
		// Draw the skeleton as a set of joint positions, connecting lines,
		// and local coordinate systems at each joint.
		// Here we'll use old style (immediate mode) OpenGL as opposed to
		// modern GL where we always first load the data to a GPU buffer.
		// Immediate mode is convenient when we want to draw a small amount
		// of things without having to write much code, and performance is
		// not an issue: for instance, when you want to want to draw something
		// for debugging.
		glUseProgram(0);
		glMatrixMode(GL_PROJECTION);
		glLoadMatrixf(&P(0,0));
		glMatrixMode(GL_MODELVIEW);
		glLoadMatrixf(&C(0,0));
		renderSkeleton();
		renderOrigin();
		renderIKpoint();
		

	} else if (drawmode_ == MODE_MESH_CPU) {
		auto vertices = computeSSD(weighted_vertices_);

		glBindBuffer(GL_ARRAY_BUFFER, gl_.simple_vertex_buffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * vertices.size(), vertices.data(), GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glUseProgram(gl_.simple_shader);
		glUniformMatrix4fv(gl_.simple_world_to_clip_uniform, 1, GL_FALSE, world_to_clip.getPtr());
		glUniform1f(gl_.simple_shading_mix_uniform, shading_toggle_ ? 1.0f : 0.0f);
		glBindVertexArray(gl_.simple_vao);
		glDrawArrays(GL_TRIANGLES, 0, (int)vertices.size());
		glBindVertexArray(0);
		glUseProgram(0);
	} else if (drawmode_ == MODE_MESH_GPU) {

		glUseProgram(0);
		glMatrixMode(GL_PROJECTION);
		glLoadMatrixf(&P(0, 0));
		glMatrixMode(GL_MODELVIEW);
		glLoadMatrixf(&C(0, 0));
		//renderSkeleton();
		renderOrigin();
		renderIKpoint();


		auto ssd_transforms = skel_.getSSDTransforms();

		glUseProgram(gl_.ssd_shader);
		glUniformMatrix4fv(gl_.ssd_world_to_clip_uniform, 1, GL_FALSE, world_to_clip.getPtr());
		glUniform1f(gl_.ssd_shading_mix_uniform, shading_toggle_ ? 1.0f : 0.0f);
		glUniformMatrix4fv(gl_.ssd_transforms_uniform, GLsizei(ssd_transforms.size()), GL_FALSE, (GLfloat*)ssd_transforms.data());
		
		glBindVertexArray(gl_.ssd_vao);
		glDrawArrays(GL_TRIANGLES, 0, (int)weighted_vertices_.size());
		glBindVertexArray(0);
		glUseProgram(0);
	}
	else if (drawmode_ == MODE_MESH_GPU_DQ)
	{
		glUseProgram(0);
		glMatrixMode(GL_PROJECTION);
		glLoadMatrixf(&P(0, 0));
		glMatrixMode(GL_MODELVIEW);
		glLoadMatrixf(&C(0, 0));
		renderOrigin();
		renderIKpoint();

		auto ssd_transforms = skel_.getSSDTransforms();
		

		// convert transforms to dual quaternions
		std::vector<FW::Vec4f> dualquaternions_0;
		std::vector<FW::Vec4f> dualquaternions_e;

		for (const auto& M : ssd_transforms)
		{
			FW::Mat3f rMat = M.getXYZ();

			FW::Vec3f c1 = rMat.col(0);
			FW::Vec3f c2 = rMat.col(1);
			FW::Vec3f c3 = rMat.col(2);

			Eigen::Matrix3f rotation;
			rotation << c1.x, c2.x, c3.x,
				c1.y, c2.y, c3.y,
				c1.z, c2.z, c3.z;

			//rotation << c1.x, c1.y, c1.z, c2.x, c2.y, c2.z, c3.x, c3.y, c3.z;

			Eigen::Quaternionf qRotation(rotation);
			qRotation.normalize();
			dualquaternions_0.push_back(Vec4f(qRotation.x(), qRotation.y(), qRotation.z(), qRotation.w()));

			FW::Vec4f t = M.col(3);
			t = t * 0.5;
			Eigen::Quaternionf qt = Eigen::Quaternionf(0.f, t.x, t.y, t.z);
			
			Eigen::Quaternionf qTranslation = qt * qRotation;

			dualquaternions_e.push_back(Vec4f(qTranslation.x(), qTranslation.y(), qTranslation.z(), qTranslation.w()));
		}

		
		glUseProgram(gl_.dq_shader);
		glUniformMatrix4fv(gl_.dq_world_to_clip_uniform, 1, GL_FALSE, world_to_clip.getPtr());
		glUniform1f(gl_.dq_shading_mix_uniform, shading_toggle_ ? 1.0f : 0.0f);
		glUniform4fv(gl_.dq_dualquaternions_0_uniform, GLsizei(dualquaternions_0.size()), (GLfloat*)dualquaternions_0.data());
		glUniform4fv(gl_.dq_dualquaternions_e_uniform, GLsizei(dualquaternions_e.size()), (GLfloat*)dualquaternions_e.data());
		
		glBindVertexArray(gl_.dq_vao);
		glDrawArrays(GL_TRIANGLES, 0, (int)weighted_vertices_.size());
		glBindVertexArray(0);
		glUseProgram(0);
	}
	
	// Check for OpenGL errors.
	GLContext::checkErrors();
	
	// Show status messages.
	common_ctrl_.message(sprintf("Ik point location: (%.2f, %.2f, %.2f)\n\n",
		IKTargetPointLocation.x, IKTargetPointLocation.y, IKTargetPointLocation.z), "IkData");
	//common_ctrl_.message(sprintf("Use Home/End to rotate camera, Q/W to change selected bone\n    Arrow keys and PgUp/Dn to rotate selected bone\n    R to reset current bone rotation"), "instructions");
	auto joint_rot = skel_.getJointRotation(selected_joint_);
	auto joint_pos = Vec4f(skel_.getToWorldTransforms()[selected_joint_].getCol(3)).getXYZ();
	common_ctrl_.message(sprintf("Joint \"%s\" selected, rotation %.2f %.2f %.2f",
						skel_.getJointName(selected_joint_).c_str(),
		joint_pos.x, joint_pos.y, joint_pos.z), "jointdata");
}

void App::renderSkeleton() {
	glEnable(GL_POINT_SMOOTH);
	glPointSize(15);
	
	// Let's fetch the transforms you generated in Skeleton::updateToWorldTransforms().
	vector<Mat4f> transforms = skel_.getToWorldTransforms();

	// And loop through all the joints.
	for (auto i = 0u; i < transforms.size(); ++i) {
		// YOUR CODE HERE (R1)
		// Use the transforms to obtain the position of the joint in world space.
		// This should be a one-liner.
		Vec3f joint_world_pos = (transforms[i] * Vec4f(0.f,0.f,0.f,1.f)).getXYZ();

		// glBegin()-glEnd() with glVertex() commands in between is how draw calls
		// are done in immediate mode OpenGL.
		glBegin(GL_POINTS);
		if (i == (int)selected_joint_)
			glColor3f( 1.0f, 0.2f, 0.2f );
		else
			glColor3f( 1.0f, 1.0f, 1.0f );
			
		//glVertex3f(0.f, 0.f, 0.f);
		// Immediate mode command drawing an individual vertex
		glVertex3f(joint_world_pos.x, joint_world_pos.y, joint_world_pos.z);
		glEnd(); // we're done drawing points

		// YOUR CODE HERE (R3)
		// Use the transforms to obtain the joint's orientation.
		// (If you understand transformation matrices correctly, you can directly
		// read the these vectors off of the matrices.)
		Vec3f right, up, ahead;
		right = Vec4f(transforms[i].getCol(0)).getXYZ();
		up = Vec4f(transforms[i].getCol(1)).getXYZ();
		ahead = Vec4f(transforms[i].getCol(2)).getXYZ();


		// Then let's draw some lines to show the joint coordinate system.
		// Draw a small coloured line segment from the joint's world position towards
		// each of its local coordinate axes (the line length should be determined by "scale").
		// The colors for each axis are already set for you below.
		const float scale = 0.05;	// length for the coordinate system axes.
		glBegin(GL_LINES);

		// draw the x axis... ("right")
		glColor3f(1, 0, 0); // red
		glVertex3f(joint_world_pos.x, joint_world_pos.y, joint_world_pos.z);
		right = joint_world_pos + right * scale;
		glVertex3f(right.x, right.y, right.z);

		// ..and the y axis.. ("up")
		glColor3f(0, 1, 0); // green
		glVertex3f(joint_world_pos.x, joint_world_pos.y, joint_world_pos.z);
		up = joint_world_pos + up * scale;
		glVertex3f(up.x, up.y, up.z);

		// ..and the z axis ("ahead").
		glColor3f(0, 0, 1); // blue
		glVertex3f(joint_world_pos.x, joint_world_pos.y, joint_world_pos.z);
		ahead = joint_world_pos + ahead * scale;
		glVertex3f(ahead.x, ahead.y, ahead.z);

		// Finally, draw a line segment from the world position of this joint to the world
		// position of the parent joint. You should first check if the parent exists
		// using skel_.getJointParent(i) - it returns -1 for the root, which has no parent.
		glColor3f(1, 1, 1);

		int parent = skel_.getJointParent(i);
		if ( parent != -1)
		{
			glVertex3f(joint_world_pos.x, joint_world_pos.y, joint_world_pos.z);
			Vec3f parent_world_pos = (transforms[parent] * Vec4f(0.f, 0.f, 0.f, 1.f)).getXYZ();
			glVertex3f(parent_world_pos.x, parent_world_pos.y, parent_world_pos.z);
		}
		// ...
		glEnd(); // we're done drawing lines	
	}
}

void FW::App::renderOrigin(void)
{

	const float scale = 0.05;	// length for the coordinate system axes.
	glBegin(GL_LINES);

	// draw the x axis... ("right")
	glColor3f(1, 0, 0); // red
	glVertex3f(0, 0, 0);
	Vec3f right =  Vec3f(1.f,0.f,0.f) * scale;
	glVertex3f(right.x, right.y, right.z);

	// ..and the y axis.. ("up")
	glColor3f(0, 1, 0); // green
	glVertex3f(0, 0, 0);
	Vec3f up = Vec3f(0.f, 1.f, 0.f) * scale;
	glVertex3f(up.x, up.y, up.z);

	// ..and the z axis ("ahead").
	glColor3f(0, 0, 1); // blue
	glVertex3f(0, 0, 0);
	Vec3f forward = Vec3f(0.f, 0.f, 1.f) * scale;
	glVertex3f(forward.x, forward.y, forward.z);

	glEnd(); // we're done drawing lines	
}

void FW::App::renderIKpoint(void)
{
	glEnable(GL_POINT_SMOOTH);
	glPointSize(15);

	glBegin(GL_POINTS);
	glColor3f(0.2f, 0.2f, 1.f);
	// Immediate mode command drawing an individual vertex

	glVertex3f(IKTargetPointLocation.x, IKTargetPointLocation.y, IKTargetPointLocation.z);

	glEnd(); // we're done drawing points
}

vector<Vertex> App::computeSSD(const vector<WeightedVertex>& source_vertices) {
	vector<Mat4f> ssd_transforms = skel_.getSSDTransforms();
	vector<Vertex> skinned_vertices;
	skinned_vertices.reserve(source_vertices.size());




	// convert transforms to dual quaternions
	std::vector<FW::Vec4f> dualquaternions_0;
	std::vector<FW::Vec4f> dualquaternions_e;

	for (const auto& M : ssd_transforms)
	{
		FW::Mat3f rMat = M.getXYZ();

		FW::Vec3f c1 = rMat.col(0);
		FW::Vec3f c2 = rMat.col(1);
		FW::Vec3f c3 = rMat.col(2);

		Eigen::Matrix3f rotation;
		rotation << c1.x, c2.x, c3.x,
			c1.y, c2.y, c3.y,
			c1.z, c2.z, c3.z;

		//rotation << c1.x, c1.y, c1.z, c2.x, c2.y, c2.z, c3.x, c3.y, c3.z;

		Eigen::Quaternionf qRotation(rotation);
		//qRotation.normalize();
		dualquaternions_0.push_back(Vec4f(qRotation.x(), qRotation.y(), qRotation.z(), qRotation.w()));

		FW::Vec4f t = M.col(3);
		//t = t * 0.5;
		Eigen::Quaternionf qt = Eigen::Quaternionf(0.f, t.x, t.y, t.z);
		//qt.normalize();
		Eigen::Quaternionf qTranslation = qt * qRotation;
		qTranslation.coeffs() = qTranslation.coeffs() * 0.5;

		dualquaternions_e.push_back(Vec4f(qTranslation.x(), qTranslation.y(), qTranslation.z(), qTranslation.w()));
	}





	
	for (const auto& sv : source_vertices) {
		// YOUR CODE HERE (R4 & R5)
		// Compute the skinned position for each vertex and push it to skinned_vertices.
		// This starter code just copies the source vertex untouched, so the result is not animated.
		// Replace these lines with the loop that applies the bone transforms and weights.
		// For R5, transform the normals as well.
		Vertex v;

		Vec4f b0 = Vec4f(0.f);
		Vec4f be = Vec4f(0.f);

		Mat4f mmm(0.f);

		for (int i = 0; i < WEIGHTS_PER_VERTEX; i++)
		{

			if (sv.weights[i] != 0)
			{

				b0 += sv.weights[i] * dualquaternions_0[sv.joints[i]];
				be += sv.weights[i] * dualquaternions_e[sv.joints[i]];

				mmm = mmm + sv.weights[i] * ssd_transforms[sv.joints[i]];
				//v.position += (sv.weights[i] * ssd_transforms[sv.joints[i]] * Vec4f(sv.position, 1)).getXYZ();
				v.normal += (sv.weights[i] * ssd_transforms[sv.joints[i]] * Vec4f(sv.normal, 0)).getXYZ();
				v.color += (sv.weights[i] * sv.color);
			}
		}




		//b0 = normalize(b0);
		//be = normalize(be);
		Vec4f c0 = b0 / b0.length();
		Vec4f ce = be / b0.length();

		float x0 = c0[0];
		float y0 = c0[1];
		float z0 = c0[2];
		float w0 = c0[3];


		float xe = ce[0];
		float ye = ce[1];
		float ze = ce[2];
		float we = ce[3];


		// denote the components of b0 as w0,x0,y0, z0
		// denote the components of be as we ,xe ,ye , ze
		float t0 = 2. * (-1. * we * x0 + xe * w0 - ye * z0 + ze * y0);
		float t1 = 2. * (-1. * we * y0 + xe * z0 + ye * w0 - ze * x0);
		float t2 = 2. * (-1. * we * z0 - xe * y0 + ye * x0 + ze * w0);

		Vec4f c1 = Vec4f(1. - 2. * y0 * y0 - 2. * z0 * z0,
			2. * x0 * y0 + 2. * w0 * z0,
			2. * x0 * z0 - 2. * w0 * y0,
			0.);

		Vec4f c2 = Vec4f(2. * x0 * y0 - 2. * w0 * z0,
			1. - 2. * x0 * x0 - 2. * z0 * z0,
			2. * y0 * z0 + 2. * w0 * x0,
			0.);

		Vec4f c3 = Vec4f(2. * x0 * z0 + 2. * w0 * y0,
			2. * y0 * z0 - 2. * w0 * x0,
			1. - 2. * x0 * x0 - 2. * y0 * y0,
			0.);

		Vec4f c4 = Vec4f(t0, t1, t2, 1.);


		Mat4f tranformMat;
		tranformMat.setCol(0, c1);
		tranformMat.setCol(1, c2);
		tranformMat.setCol(2, c3);
		tranformMat.setCol(3, c4);





		v.position = (tranformMat * Vec4f(sv.position, 1)).getXYZ();





		skinned_vertices.push_back(v);
	}
	return skinned_vertices;
}

void FW::App::SelectBoneByMouse(const Window::Event& ev)
{



	// Something close to the debugging part of the subdiv assignment
	Vec2f mouse_pos_ = Vec2f(ev.mousePos) / Vec2f(window_.getSize()) * 2.0f - 1.0f;
	mouse_pos_.y *= -1;

	Mat4f objectToCamera, projection;
	glGetFloatv(GL_MODELVIEW_MATRIX, objectToCamera.getPtr());
	glGetFloatv(GL_PROJECTION_MATRIX, projection.getPtr());

	Mat4f world_to_clip = projection * objectToCamera;

	vector<Mat4f> transforms = skel_.getToWorldTransforms();

	float minCost = 1e9;
	int closestId = -1;

	// And loop through all the joints.
	for (auto i = 0u; i < transforms.size(); ++i) {
		
		Vec3f joint_world_pos = (transforms[i] * Vec4f(0.f, 0.f, 0.f, 1.f)).getXYZ();
		Vec4f clip = world_to_clip * Vec4f( joint_world_pos, 1.f);
		Vec3f clipPos = clip.getXYZ() / clip.w;
		float depth = clip.w;

		// use a cost function that prefers points that are closer to camera
		float dist = (clipPos.getXY() - mouse_pos_).length();
		float cost = dist + depth * .01f;

		if (cost < minCost)
		{
			minCost = cost;
			closestId = i;
		}
	}

	if (closestId != -1)
		selected_joint_ = closestId;
	
}

void FW::App::FindPointInCameraXYplane(int axis)
{
	Vec3f newPos = FindMouseToWorld(clip_mouse_pos_);
	IKTargetPointLocation[axis] += newPos[axis] - init_mouse_to_world_pos_[axis];
	init_mouse_to_world_pos_[axis] = newPos[axis];
	
}

Vec3f FW::App::FindMouseToWorld(Vec2f mousePos)
{

	Mat4f objectToCamera, projection;
	glGetFloatv(GL_MODELVIEW_MATRIX, objectToCamera.getPtr());
	glGetFloatv(GL_PROJECTION_MATRIX, projection.getPtr());

	vector<Mat4f> transforms = skel_.getToWorldTransforms();
	Mat4f clip_to_world = (transforms[selected_joint_] * projection * objectToCamera).inverted();
	
	float zmouse;
	glReadPixels(mousePos.x, mousePos.y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &zmouse);

	Vec4f mousepos = clip_to_world * Vec4f(mousePos, zmouse, 1.f);
	mousepos /= mousepos.w;
	

	return mousepos.getXYZ();
}

vector<WeightedVertex> App::loadAnimatedMesh(string namefile, string mesh_file, string attachment_file) {
	vector<WeightedVertex> vertices;
	vector<array<int, WEIGHTS_PER_VERTEX>> indices;
	vector<array<float, WEIGHTS_PER_VERTEX>> weights;
	vector<Vec3f> colors;
	
	// Load name to index conversion map
	vector<string> names;
	ifstream name_input(namefile, ios::in);
	string name;
	while (name_input.good())
	{
		name_input >> name;
		names.push_back(name);
	}

	// Load vertex weights
	ifstream attachment_input(attachment_file, ios::in);
	string line;
	while (getline(attachment_input, line)) {
		auto temp_i = array<int, WEIGHTS_PER_VERTEX>();
		auto temp_w = array<float, WEIGHTS_PER_VERTEX>();
		stringstream ss(line);
		int sink;
		ss >> sink;
		auto n_weights = 0u;
		while(ss.good()) {
			string word;
			ss >> word;
			int i = stoi(word.substr(0, word.find_first_of('.')));
			float weight;
			ss >> weight;
			if (weight != 0) {
				temp_w[n_weights] = weight;
				temp_i[n_weights] = skel_.getJointIndex(names[names.size()-i-2]);
				++n_weights;
			}
		}
		assert(n_weights <= WEIGHTS_PER_VERTEX);
		weights.push_back(temp_w);
		indices.push_back(temp_i);
		auto color = Vec3f();
		for (auto i = 0u; i < WEIGHTS_PER_VERTEX; ++i)
			color += joint_colors_[temp_i[i]] * temp_w[i];
		colors.push_back(color);
	}

	// Load vertices
	vector<Vec3f> positions;
	ifstream mesh_input(mesh_file, ios::in);
	while (getline(mesh_input, line)) {
		stringstream ss(line);
		string s;
		ss >> s;
		if (s == "v") {
			Vec3f pos;
			ss >> pos[0] >> pos[1] >> pos[2];
			positions.push_back(pos);
		}
		else if (s == "f") {
			array<int, 3> f;

			string word;
			for (int i = 0; i < 3; ++i)
			{
				ss >> word;
				word = word.substr(0, word.find_first_of('/'));
				f[i] = stoi(word) - 1;
			}

			WeightedVertex v;
			v.normal = cross(positions[f[1]] - positions[f[0]], positions[f[2]] - positions[f[0]]).normalized();
			for (auto i : f) {
				v.position = positions[i];
				v.color = colors[i];
				memcpy(v.joints, indices[i].data(), sizeof(v.joints));
				memcpy(v.weights, weights[i].data(), sizeof(v.weights));
				vertices.push_back(v);
			}

			// Read n-gons
			while (ss.good())
			{
				f[1] = f[2];
				ss >> word;
				word = word.substr(0, word.find_first_of('/'));
				f[2] = stoi(word) - 1;

				v.normal = cross(positions[f[1]] - positions[f[0]], positions[f[2]] - positions[f[0]]).normalized();
				for (auto i : f) {
					v.position = positions[i];
					v.color = colors[i];
					memcpy(v.joints, indices[i].data(), sizeof(v.joints));
					memcpy(v.weights, weights[i].data(), sizeof(v.weights));
					vertices.push_back(v);
				}
			}
		}
	}

	// Normalize mesh scale
	//scale = 0;
	//for (auto& v : vertices)
	//	scale = max(scale, abs(v.position).max());
	for (auto& v : vertices)
		v.position = (v.position - Vec3f(0, 97, 0)) / scale_;

	int idx = 0;
	for (const auto& v : vertices) {
		float sum_weights = accumulate(begin(v.weights), end(v.weights), 0.0f);
		(void)sum_weights; // silence warning on release build
		assert(0.99 < sum_weights && sum_weights < 1.01 && "weights do not sum up to 1");
		for (auto i = 0u; i < WEIGHTS_PER_VERTEX; ++i) {
			assert(0 <= v.joints[i] && "invalid index");
		}
		idx++;
	}
	return vertices;
}

vector<WeightedVertex> App::loadWeightedMesh(string mesh_file, string attachment_file) {
	vector<WeightedVertex> vertices;
	vector<array<int, WEIGHTS_PER_VERTEX>> indices;
	vector<array<float, WEIGHTS_PER_VERTEX>> weights;
	vector<Vec3f> colors;
	ifstream attachment_input(attachment_file, ios::in);
	string line;
	while(getline(attachment_input, line)) {
		auto temp_i = array<int, WEIGHTS_PER_VERTEX>();
		auto temp_w = array<float, WEIGHTS_PER_VERTEX>();
		stringstream ss(line);
		auto n_weights = 0u;
		for (auto i = 0u; ss.good(); ++i) {
			float weight;
			ss >> weight;
			if (weight != 0) {
				temp_w[n_weights] = weight;
				temp_i[n_weights] = i+1;
				++n_weights;
			}
		}
		assert(n_weights <= WEIGHTS_PER_VERTEX);

		weights.push_back(temp_w);
		indices.push_back(temp_i);
		auto color = Vec3f();
		for (auto i = 0u; i < WEIGHTS_PER_VERTEX; ++i)
		{
			color += joint_colors_[temp_i[i]] * temp_w[i];

		}
		colors.push_back(color);
	}

	vector<Vec3f> positions;
	ifstream mesh_input(mesh_file, ios::in);
	while(getline(mesh_input, line)) {
		stringstream ss(line);
		string s;
		ss >> s;
		if (s == "v") {
			Vec3f pos;
			ss >> pos[0] >> pos[1] >> pos[2];
			positions.push_back(pos);
		} else if (s == "f") {
			array<int, 3> f;
			ss >> f[0] >> f[1] >> f[2];
			for (auto& i : f) --i;
			WeightedVertex v;
			v.normal = cross(positions[f[1]]-positions[f[0]], positions[f[2]]-positions[f[0]]).normalized();
			for (auto i : f) {
				v.position = positions[i];
				v.color = colors[i];
				memcpy(v.joints, indices[i].data(), sizeof(v.joints));
				memcpy(v.weights, weights[i].data(), sizeof(v.weights));
				vertices.push_back(v);
			}
		}
	}

	for (auto& v : vertices) {
		

		
		int LeftHandIndex = skel_.getJointIndex("Left_hand");
		int LeftElbowIndex = skel_.getJointIndex("Left_elbow");

		joint_colors_[19] = Vec3f(1.f, 0.f, 0.f);
		joint_colors_[14] = Vec3f(0.f, 0.f, 1.f);


		auto attachedToLeftElbow = std::find(begin(v.joints), end(v.joints), LeftElbowIndex);
		vector<Mat4f> transforms = skel_.getToWorldTransforms();
		vector<Mat4f> ssd_transforms = skel_.getSSDTransforms();
		if (attachedToLeftElbow != end(v.joints)) // this vertex is attached to left elbow
		{
			//std::cout << "---------------------\n";
			float LECurrentWeight = v.weights[attachedToLeftElbow - begin(v.joints)];
			Vec3f LEjoint_world_pos = (transforms[LeftElbowIndex] * Vec4f(0.f, 0.f, 0.f, 1.f)).getXYZ();
			//std::cout << "LEjoint_world_pos: "; LEjoint_world_pos.print(); std::cout << endl;
			Vec3f LHjoint_world_pos = (transforms[LeftHandIndex] * Vec4f(0.f, 0.f, 0.f, 1.f)).getXYZ();

			LHjoint_world_pos += (LHjoint_world_pos - LEjoint_world_pos).normalized() * 0.085f;

			//std::cout << "LHjoint_world_pos: "; LHjoint_world_pos.print(); std::cout << endl;

			//std::cout << "position: "; v.position.print(); std::cout << endl;

			float distToLE = (v.position - LEjoint_world_pos).length();
			float distToLH = (v.position - LHjoint_world_pos).length();

			float wLH = (distToLE / (distToLE + distToLH));
			float wLE = (distToLH / (distToLE + distToLH));
			//std::cout << "wLH: " << wLH; std::cout << endl;
			//std::cout << "wLE: "<< wLE; std::cout << endl;

			wLH *= LECurrentWeight;
			wLE *= LECurrentWeight;

			auto emptySpot = std::find(begin(v.weights), end(v.weights), 0);

			if (emptySpot != end(v.weights))
			{

				if (wLH > 0.65 * LECurrentWeight)
				{
					wLH += wLE;
					wLE = 0;
				}
				if (!(wLH < 0.55 * LECurrentWeight))
				{
					v.weights[attachedToLeftElbow - begin(v.joints)] = wLE;
					//std::cout << attachedToLeftElbow - begin(v.joints) << std::endl;
					v.weights[emptySpot - begin(v.weights)] = wLH;

					v.joints[emptySpot - begin(v.weights)] = LeftHandIndex;
				}
				
			}
		}

		auto color = Vec3f();
		for (auto i = 0u; i < WEIGHTS_PER_VERTEX; ++i)
			color += joint_colors_[v.joints[i]] * v.weights[i];
		v.color = color;



		float sum_weights = accumulate(begin(v.weights), end(v.weights), 0.0f);
		(void)sum_weights; // silence warning on release build
		assert(0.99 < sum_weights && sum_weights < 1.01 && "weights do not sum up to 1");
		for (auto i = 0u; i < WEIGHTS_PER_VERTEX; ++i) {

			assert(0 <= v.joints[i] && "invalid index");
		}
	}
	return vertices;
}

void App::loadAnimation(const String& filename) {
	int end = filename.lastIndexOf('.');
	String prefix = (end > 0) ? filename.substring(0, end) : filename;
	string p(prefix.getPtr());
	string skel_file = p + ".bvh";
	string mesh_file = p + ".bvhobj";
	string weight_file = p + ".weights";
	string name_file = p + ".names";

	cout << "skeleton:   " << skel_file << endl;
	cout << "mesh:       " << mesh_file << endl;
	cout << "weight:     " << weight_file << endl;

	scale_ = skel_.loadBVH(skel_file); 
	weighted_vertices_ = loadAnimatedMesh(name_file, mesh_file, weight_file);
}

void App::loadModel(const String& filename) {
	int end = filename.lastIndexOf('.');
	String prefix = (end > 0) ? filename.substring(0, end) : filename;
	string p(prefix.getPtr());
	string skel_file = p + ".skel";
	string mesh_file = p + ".obj";
	string weight_file = p + ".attach";

	cout << "skeleton:   " << skel_file << endl;
	cout << "mesh:       " << mesh_file << endl;
	cout << "weight:     " << weight_file << endl;

	scale_ = 1;
	skel_.load(skel_file);
	weighted_vertices_ = loadWeightedMesh(mesh_file, weight_file);
}

void FW::init(void) {
	new App;
}
