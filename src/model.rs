use crate::device::BfDevice;

use std::hash::{Hash, Hasher};
use std::mem::size_of;
use std::ptr::copy_nonoverlapping as memcpy;

use anyhow::Result;
//use cgmath::{vec2, vec3, point3, Deg};
use vulkanalia::prelude::v1_0::*;

type Vec2 = cgmath::Vector2<f32>;
type Vec3 = cgmath::Vector3<f32>;
//type Mat4 = cgmath::Matrix4<f32>;
//type Pt3  = cgmath::Point3<f32>;

pub struct Model {
    pub vertex_buffer: vk::Buffer,
    pub vertex_buffer_memory: vk::DeviceMemory,
}

impl Model {

    pub unsafe fn new(bf_device: &BfDevice, vertices: Vec<Vertex>) -> Result<Self> {
        let (vertex_buffer, vertex_buffer_memory) = Self::create_vertex_buffer(&bf_device, vertices)?;

        Ok(Self {
            vertex_buffer,
            vertex_buffer_memory,
        })
    }

    unsafe fn create_vertex_buffer(
        bf_device: &BfDevice,
        vertices: Vec<Vertex>,
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let size = (size_of::<Vertex>() * vertices.len()) as u64;

        let (staging_buffer, staging_buffer_memory) = bf_device.create_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        let memory = bf_device.device.map_memory(
            staging_buffer_memory,
            0,
            size,
            vk::MemoryMapFlags::empty(),
        )?;

        memcpy(vertices.as_ptr(), memory.cast(), vertices.len());

        bf_device.device.unmap_memory(staging_buffer_memory);

        let (vertex_buffer, vertex_buffer_memory) = bf_device.create_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        bf_device.copy_buffer(staging_buffer, vertex_buffer, size)?;

        bf_device.device.destroy_buffer(staging_buffer, None);
        bf_device.device.free_memory(staging_buffer_memory, None);

        Ok((vertex_buffer, vertex_buffer_memory))
    }

}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pos: Vec3,
    color: Vec3,
    tex_coord: Vec2,
}

impl Vertex {
    pub const fn new(pos: Vec3, color: Vec3, tex_coord: Vec2) -> Self {
        Self { pos, color, tex_coord }
    }

    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }

    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        let pos = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0)
            .build();

        let color = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(size_of::<Vec3>() as u32)
            .build();

        let tex_coord = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32_SFLOAT)
            .offset((size_of::<Vec3>() + size_of::<Vec3>()) as u32)
            .build();

        [pos, color, tex_coord]
    }
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos
            && self.color == other.color
            && self.tex_coord == other.tex_coord
    }
}

impl Eq for Vertex {}

impl Hash for Vertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pos[0].to_bits().hash(state);
        self.pos[1].to_bits().hash(state);
        self.pos[2].to_bits().hash(state);
        self.color[0].to_bits().hash(state);
        self.color[1].to_bits().hash(state);
        self.color[2].to_bits().hash(state);
        self.tex_coord[0].to_bits().hash(state);
        self.tex_coord[1].to_bits().hash(state);
    }
}
