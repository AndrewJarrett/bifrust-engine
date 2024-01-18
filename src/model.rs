use crate::device::BfDevice;

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::mem::size_of;
use std::ptr::copy_nonoverlapping as memcpy;
use std::io::BufReader;
use std::fs::File;

use anyhow::Result;
use cgmath::{vec2, vec3};
use vulkanalia::prelude::v1_0::*;

type Vec2 = cgmath::Vector2<f32>;
type Vec3 = cgmath::Vector3<f32>;
//type Mat4 = cgmath::Matrix4<f32>;
//type Pt3  = cgmath::Point3<f32>;

pub struct Model {
    pub vertex_buffer: vk::Buffer,
    pub vertex_buffer_memory: vk::DeviceMemory,
    pub index_buffer: vk::Buffer,
    pub index_buffer_memory: vk::DeviceMemory,
    pub builder: ModelBuilder,
}

impl Model {

    pub fn new(bf_device: &BfDevice, builder: ModelBuilder) -> Result<Self> {
        let mut vertex_buffer: vk::Buffer = Default::default();
        let mut vertex_buffer_memory: vk::DeviceMemory = Default::default();
        let mut index_buffer: vk::Buffer = Default::default();
        let mut index_buffer_memory: vk::DeviceMemory = Default::default();

        unsafe {
            (vertex_buffer, vertex_buffer_memory) = Self::create_vertex_buffer(&bf_device, &builder.vertices)?;

            (index_buffer, index_buffer_memory) = Self::create_index_buffer(&bf_device, &builder.indices)?;
        }

        Ok(Self {
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,
            builder,
        })
    }

    unsafe fn create_vertex_buffer(
        bf_device: &BfDevice,
        vertices: &Vec<Vertex>,
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

    unsafe fn create_index_buffer(
        bf_device: &BfDevice,
        indices: &Vec<u32>,
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let size = (size_of::<u32>() * indices.len()) as u64;

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

        memcpy(indices.as_ptr(), memory.cast(), indices.len());

        bf_device.device.unmap_memory(staging_buffer_memory);

        let (index_buffer, index_buffer_memory) = bf_device.create_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        bf_device.copy_buffer(staging_buffer, index_buffer, size)?;

        bf_device.device.destroy_buffer(staging_buffer, None);
        bf_device.device.free_memory(staging_buffer_memory, None);

        Ok((index_buffer, index_buffer_memory))
    }

    pub unsafe fn draw(
        &self,
        bf_device: &BfDevice,
        command_buffer: vk::CommandBuffer
    ) -> Result<()> {
        bf_device.device.cmd_draw_indexed(command_buffer, self.builder.indices.len() as u32, 1, 0, 0, 0);

        Ok(())
    }

    pub unsafe fn bind(
        &self,
        bf_device: &BfDevice,
        command_buffer: vk::CommandBuffer
    ) -> Result<()> {
        bf_device.device.cmd_bind_vertex_buffers(
            command_buffer,
            0,
            &[self.vertex_buffer],
            &[0]
        );
        bf_device.device.cmd_bind_index_buffer(
            command_buffer,
            self.index_buffer,
            0,
            vk::IndexType::UINT32
        );
        
        Ok(())
    }

}

#[derive(Debug)]
pub struct ModelBuilder {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl ModelBuilder {
    
    pub fn new() -> Result<Self> {
        let mut reader = BufReader::new(File::open("resources/viking_room.obj")?);
        //let mut reader = BufReader::new(File::open("resources/orange-3d.obj")?);
        
        let mut indices: Vec<u32> = Vec::new();
        let mut vertices: Vec<Vertex> = Vec::new();

        let (models, _) = tobj::load_obj_buf(
            &mut reader,
            &tobj::LoadOptions { triangulate: true, ..Default::default() },
            |_| Ok(Default::default()),
        )?;

        let mut unique_vertices = HashMap::new();

        for model in &models {
            for index in &model.mesh.indices {
                let pos_offset = (3 * index) as usize;
                let tex_coord_offset = (2 * index) as usize;

                let vertex = Vertex {
                    pos: vec3(
                        model.mesh.positions[pos_offset],
                        model.mesh.positions[pos_offset + 1],
                        model.mesh.positions[pos_offset + 2],
                    ),
                    color: vec3(1.0, 1.0, 1.0),
                    tex_coord: vec2(
                        model.mesh.texcoords[tex_coord_offset],
                        1.0 - model.mesh.texcoords[tex_coord_offset + 1],
                    ),
                };

                if let Some(index) = unique_vertices.get(&vertex) {
                    indices.push(*index as u32);
                } else {
                    let index = vertices.len();
                    unique_vertices.insert(vertex, index);
                    vertices.push(vertex);
                    indices.push(index as u32);
                }
            }
        }

        Ok(Self {
            indices,
            vertices
        })
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
