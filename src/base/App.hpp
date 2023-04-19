#pragma once

#include "skeleton.hpp"

#include "gui/Window.hpp"
#include "gui/CommonControls.hpp"

#include <vector>

namespace FW {

struct Vertex
{
	Vec3f position;
	Vec3f normal;
	Vec3f color;
};

struct WeightedVertex
{
	Vec3f	position;
	Vec3f	normal;
	Vec3f	color;
	int		joints[WEIGHTS_PER_VERTEX];
	float	weights[WEIGHTS_PER_VERTEX];
};

struct glGeneratedIndices
{
	// Shader programs
	GLuint simple_shader, ssd_shader, dq_shader;

	// Vertex array objects
	GLuint simple_vao, ssd_vao, dq_vao;

	// Buffers
	GLuint simple_vertex_buffer, ssd_vertex_buffer, dq_vertex_buffer;

	// simple_shader uniforms
	GLint simple_world_to_clip_uniform, simple_shading_mix_uniform;

	// ssd_shader uniforms
	GLint ssd_world_to_clip_uniform, ssd_shading_mix_uniform, ssd_transforms_uniform;

	// dq_shader uniforms
	GLint dq_world_to_clip_uniform, dq_shading_mix_uniform, dq_dualquaternions_0_uniform, dq_dualquaternions_e_uniform;
};

class App : public Window::Listener
{
private:
	enum DrawMode
	{
		MODE_SKELETON,
		MODE_MESH_CPU,
		MODE_MESH_GPU,
		MODE_MESH_GPU_DQ
	};

public:
					App             (void);
	virtual         ~App            (void) {}

	virtual bool    handleEvent     (const Window::Event& ev);

private:
	void			initRendering		(void);
	void			render				(void);
	void			renderSkeleton		(void);
	void			renderOrigin(void);
	void			renderIKpoint(void);

	void			loadModel			(const String& filename);
	void			loadAnimation		(const String& filename);
	

	std::vector<WeightedVertex>	loadAnimatedMesh		(std::string namefile, std::string mesh_file, std::string attachment_file);
	std::vector<WeightedVertex>	loadWeightedMesh		(std::string mesh_file, std::string attachment_file);

	std::vector<Vertex>			computeSSD				(const std::vector<WeightedVertex>& source);


	void						SelectBoneByMouse		(const Window::Event& ev);
	void						FindPointInCameraXYplane(int axis);
	Vec3f						FindMouseToWorld(Vec2f mousePos);
private:
					App             (const App&); // forbid copy
	App&            operator=       (const App&); // forbid assignment

private:
	Window			window_;
	CommonControls	common_ctrl_;

	DrawMode		drawmode_;
	String			filename_;
	bool			shading_toggle_;
	bool			solve_IK_;
	bool			solve_IK_changed;
	bool			shading_mode_changed_;
	bool			take_snapshot_;
	bool			take_snapshot_changed;


	std::vector<Vec3f> joint_colors_;

	glGeneratedIndices	gl_;

	std::vector<WeightedVertex> weighted_vertices_;
	
	float			camera_rotation_;
	float			scale_ = 1.f;

	Skeleton		skel_;

	unsigned		selected_joint_;
	//Vec3f			selected_joint_pos;

	bool			animationMode = false;
	bool			animationMode_changed = false;
	DWORD			animationStart;

	bool			KEY_X_Hold;
	bool			KEY_Y_Hold;
	bool			KEY_Z_Hold;




	Vec3f			init_mouse_to_world_pos_;
	Vec2f			clip_mouse_pos_;
	Vec3f			IKTargetPointLocation;
	//Mat4f			world_to_clip;



	Timer timer_;
};

} // namespace FW
