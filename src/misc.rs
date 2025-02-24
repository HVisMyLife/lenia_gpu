#![allow(dead_code)]
use serde::{Deserialize, Serialize};
use std::fmt;
use arrayfire::*;

pub trait Cycle {
    fn next(&mut self);
    fn previous(&mut self);
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Shape {
    GaussianBump, // width, offset
    GaussianBumpMulti,
    ExponentialDecay,
    SmoothTransition,
}

impl Cycle for Shape {
    fn next(&mut self) {
        *self = match self {
            Shape::GaussianBump => Shape::GaussianBumpMulti,
            Shape::GaussianBumpMulti => Shape::ExponentialDecay,
            Shape::ExponentialDecay => Shape::SmoothTransition,
            Shape::SmoothTransition => Shape::GaussianBump,
        }
    }
    fn previous(&mut self) {
        *self = match self {
            Shape::GaussianBump => Shape::SmoothTransition,
            Shape::GaussianBumpMulti => Shape::GaussianBump,
            Shape::ExponentialDecay => Shape::GaussianBumpMulti,
            Shape::SmoothTransition => Shape::ExponentialDecay,
        }
    }
}
impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let l = match self {
            Self::GaussianBump => "Gaussian Bump",
            Self::GaussianBumpMulti => "Gaussian Bump Multi",
            Self::ExponentialDecay => "Exponential Decay",
            Self::SmoothTransition => "Smooth Transition",
        };
        write!(f, "{}", l)
    }
}
impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let l = match self {
            Self::GaussianBump => "Gaussian Bump",
            Self::GaussianBumpMulti => "Gaussian Bump Multi",
            Self::ExponentialDecay => "Exponential Decay",
            Self::SmoothTransition => "Smooth Transition",
        };
        write!(f, "{}", l)
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct Function {
    pub shape: Shape,
    pub centering: bool,  // should it be centered at x (moved down)
    pub parameters: Vec<f32>,
    pub hard_clip: bool,
}
impl fmt::Debug for Function {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut l = "".to_string();
        l += &format!("Shape: {:?}", self.shape);
        l += &format!(" | HC={}", self.hard_clip);
        l += &format!(", C={}", self.centering);
        l += &format!(", P={:?}", self.parameters);
        write!(f, "{}", l)
    }
}
impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut l = "".to_string();
        l += &format!("{}", self.shape);
        if self.hard_clip { l += &format!("\nHard clipped") }
        else { l += &format!("\nSigmoid clipped") }
        if self.centering { l += &format!("\nCentered") }
        else { l += &format!("\nNot centered") }
        //l += &format!("\nParameters: {:?}", self.parameters);
        write!(f, "{}", l)
    }
}
impl Function {
    pub fn new(shape: Shape, centering: bool, parameters: Vec<f32>, hard_clip: bool) -> Self {
        Function {
            shape,
            centering,
            parameters,
            hard_clip
        }
    }

    pub fn calc_array(&self, x: &Array<f32>) -> Array<f32> {
        // 0 - width, 1 - offset
        let mut y = match self.shape {
            Shape::GaussianBump => {
                let mut t = x - self.parameters[1];
                t = &t / self.parameters[0];
                t = &t * &t;
                t = &(-t) / 2_f32;
                t = exp(&t);
                t
            },
            Shape::GaussianBumpMulti => {
                let mut base = constant(0_f32, x.dims());
                self.parameters.chunks(3).for_each(|p| {
                    let mut t = x - p[1];
                    t = &t / p[0];
                    t = &t * &t;
                    t = &(-t) / 2_f32;
                    t = exp(&t) * p[2];
                    base = &base + &t;
                } );
                base
            },
            Shape::ExponentialDecay => {
                let mut t = x - self.parameters[1];
                t = &t / self.parameters[0];
                t = exp(&(-t));
                t = clamp(&t, &0_f32, &1_f32, false);
                t
            },
            Shape::SmoothTransition => {
                let mut t = x - self.parameters[1];
                t = &t / self.parameters[0];
                t = exp(&t);
                t = &t + 1_f32;
                t = 1_f32 / &t;
                t
            },
        };

        y = if self.hard_clip {clamp( &y, &0_f32, &1_f32, false)} else {
            let mut t = &y - 0.5_f32;
            t = &t * (-4_f32);
            t = 1_f32 + exp(&t);
            t = 1_f32 / &t;
            t
        };
        
        if self.centering { y = &y - 0.5_f32; }

        y
    }
    pub fn _calc(&self, x: f32) -> f32 {
        // 0 - width, 1 - offset
        let mut y = match self.shape {
            Shape::GaussianBump => {
                let mut t = x - self.parameters[1];
                t = t / self.parameters[0];
                t = t * t;
                t = (-t) / 2_f32;
                t = t.exp();
                t
            },
            Shape::GaussianBumpMulti => {
                self.parameters.chunks(3).map(|p| {
                    let mut t = x - p[1];
                    t = t / p[0];
                    t = t * t;
                    t = (-t) / 2_f32;
                    t = t.exp() * p[2];
                    t
                } ).sum()
            },
            Shape::ExponentialDecay => { // comes from infinity, so have to be clamped
                let mut t = x - self.parameters[1];
                t = t / self.parameters[0];
                t = (-t).exp();
                t = t.clamp(0_f32, 1_f32);
                t
            },
            Shape::SmoothTransition => {
                let mut t = x - self.parameters[1];
                t = t / self.parameters[0];
                t = t.exp();
                t = t + 1_f32;
                t = 1_f32 / t;
                t
            },
        };

        y = if self.hard_clip {y.clamp(0., 1.)} else {
            let mut t = y - 0.5_f32;
            t = t * (-4_f32);
            t = 1_f32 + t.exp();
            t = 1_f32 / t;
            t
        };
        if self.centering { y -= 0.5; }

        y
    }
}


pub struct FrameTimeAnalyzer {
    frame: Vec<f32>,
    s_time: f32,
}

impl FrameTimeAnalyzer {
    pub fn new(length: usize) -> Self {
        FrameTimeAnalyzer {
            frame: vec![0.; length],
            s_time: 0.,
        }
    }

    pub fn add_frame_time(&mut self, time: f32) {
        self.frame.pop();
        self.frame.insert(0, time);
    }

    pub fn smooth_frame_time(&mut self) -> &f32 {
        self.s_time = self.frame.iter().sum::<f32>() / (self.frame.len() as f32);
        &self.s_time
    }
}
