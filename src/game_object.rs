use crate::model::Model;

use anyhow::Result;
use cgmath::vec3;

type Vec3 = cgmath::Vector3<f32>;
//type Mat4 = cgmath::Matrix4<f32>;

static mut CURRENT_ID: usize = 0;

#[derive(Debug)]
pub struct GameObject {
    pub id: usize,
    pub model: Model,
    pub transform: TransformComponent,
    pub color: Vec3,
}

impl GameObject {
    
    pub fn new(model: Model, transform: TransformComponent) -> Result<Self> {
        let id: usize;
        unsafe {
            id = CURRENT_ID + 1;
        }

        let color = vec3::<f32>(0.0, 0.0, 0.0);

        Ok(Self {
            id,
            model,
            transform,
            color,
       })
    }

}

#[derive(Debug)]
pub struct TransformComponent {
    pub translation: Vec3,
    pub scale: Vec3,
    pub rotation: f32,
}

impl Default for TransformComponent {
    fn default() -> Self {
        let translation = vec3::<f32>(0.0, 0.0, 0.0);
        let scale = vec3::<f32>(1.0, 1.0, 1.0);
        let rotation = 0.0;

        Self {
            translation,
            scale,
            rotation,
        }
    }
}


