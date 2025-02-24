use std::{collections::HashMap, fmt};
use arrayfire::*;
use serde::{Deserialize, Serialize};
use crate::{Channel, Layer};



#[derive(Clone, Serialize, Deserialize)]
pub struct Lenia {
    pub channels: HashMap<usize, Channel>,
    pub layers: HashMap<usize, Layer>,
    pub delta: f32,
    pub fitness: f32,  // f>0.25 full; 0>f>0.1 life
    pub img: Array<f32>,
}
impl Lenia {
    pub fn new(delta: f32, channels: HashMap<usize, Channel>, layers: HashMap<usize, Layer>) -> Self {
        Self {img: Array::new_empty(Dim4::new(&[1,1,1,1])), 
            channels, layers, 
            delta, 
            fitness: 0.
        }
    }

    pub fn init(&mut self) {
        self.layers.values_mut().for_each(|l|{
            l.generate_kernel_lookup();
        });
    }
    
    pub fn generate_image(&mut self) {
        let matrix = &self.channels.values().next().unwrap().matrix;
        let r = color_bump(matrix, 1.);
        let g = color_bump(matrix, 1.5);
        let b = color_bump(matrix, 2.1);
        self.img = join_many(2, vec![&r,&g,&b]);
        
        fn color_bump(x: &Array<f32>, offset: f32) -> Array<f32> {
            let mut t = x * 3_f32;
            t = &t - offset;
            t = &t * &t;
            t = &(-t) + 1_f32;
            t
        }
    }

    pub fn evaluate(&mut self) {
        self.layers.values_mut().for_each(|l|{
            l.run(self.channels.get(&l.source_key).unwrap());
        });

        self.channels.values_mut().for_each(|ch|{
            // sum layers outputs
            let dims = self.layers.get( ch.weights.keys().next().unwrap() ).unwrap().matrix_out.dims();
            ch.matrix_out = Array::new_empty(dims);

            ch.weights.iter().for_each(|(k,w)|{
                let t = &self.layers.get(k).unwrap().matrix_out * (*w);
                ch.matrix_out = &ch.matrix_out + t; // add to output matrix
            });

            ch.finish(self.delta);
        });

        //// calculate fitness
        self.fitness = 0.;
        self.channels.values().for_each(|ch|{
            let (sum, _) = sum_all(&ch.matrix);
            let mean = sum / ch.matrix.elements() as f32;
            self.fitness += mean;
        });
        self.fitness /= self.channels.len() as f32;
    }

}



impl PartialEq for Lenia {
    fn eq(&self, other: &Self) -> bool {
        self.channels == other.channels &&
        self.layers == other.layers &&
        self.delta == other.delta
    }
}
impl fmt::Debug for Lenia {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut l = "<<<ECO>>>".to_string();
        l += "\nCh: \n";
        l += &format!("{:?}", self.channels);
        l += "\nL: \n";
        l += &format!("{:?}", self.layers);
        l += "\nDelta: ";
        l += &self.delta.to_string();
        l += "\nFitness: ";
        l += &self.fitness.to_string();
        write!(f, "{}", l)
    }
}

