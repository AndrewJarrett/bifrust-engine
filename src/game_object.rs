use crate::model::Model;

use anyhow::Result;
use cgmath::vec3;

type Vec3 = cgmath::Vector3<f32>;
type Mat4 = cgmath::Matrix4<f32>;

static mut CURRENT_ID: usize = 0;

pub struct GameObject {
    pub id: usize,
    pub model: Model,
    //pub transform: TransformComponent,
    pub color: Vec3,
}

impl GameObject {
    
    pub fn new(model: Model) -> Result<Self> {
        let id: usize;
        unsafe {
            id = CURRENT_ID + 1;
        }

        let color = vec3::<f32>(0.0, 0.0, 0.0);
        //let transform = TransformComponent::new();

        Ok(Self {
            id,
            model,
            color,
       })
    }

}

pub struct TransformComponent {
    pub translation: Vec3,
    pub scale: Vec3,
    pub rotation: f32,
    pub model: Mat4,
}

impl TransformComponent {

    pub fn new(translation: Vec3, scale: Vec3, rotation: f32, model: Mat4) -> Result<Self> {
        //let translation = vec3(0.0;

        Ok(Self {
            translation,
            scale,
            rotation,
            model,
        })
    }

}


