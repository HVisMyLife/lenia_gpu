use std::{collections::HashMap, fmt};
use arrayfire::*;
use serde::{Deserialize, Serialize};
use crate::Function;




#[derive(Clone, Serialize, Deserialize)]
pub struct Layer {
    pub kernel: Function,
    pub kernel_lookup: Array<f32>,
    pub growth_map: Function,
    pub source_key: usize, // number of channel that will be used as input
    pub matrix_out: Array<f32>,
    pub radius: usize,
}
impl Layer {
    pub fn new(
        kernel: Function,
        growth_map: Function,
        source_key: usize,
        radius: usize
    ) -> Self {
        Layer { 
            kernel, 
            kernel_lookup: Array::<f32>::new_empty( Dim4::new(&[radius as u64*2+1, radius as u64*2+1, 1, 1]) ),
            growth_map, source_key, matrix_out: Array::<f32>::new_empty(Dim4::new(&[512, 512, 1, 1])), radius,
        }
    }

    pub fn generate_kernel_lookup(&mut self) {
        let mut kernel_lookup = vec![0.; (self.radius*2+1)*(self.radius*2+1)];
        let h = (self.radius * 2 + 1) as i64;
        for x in -(self.radius as i64)..=self.radius as i64 {
            for y in -(self.radius as i64)..=self.radius as i64 {
                let r = ( (x*x+y*y) as f32 ).sqrt();
                kernel_lookup[((y+self.radius as i64) * h + (x + self.radius as i64)) as usize] 
                    = r/self.radius as f32;
            }    
        }
        self.kernel_lookup = Array::<f32>::new(&kernel_lookup, 
            Dim4::new(&[self.radius as u64 *2+1, self.radius as u64 *2+1, 1, 1]));
        self.kernel_lookup = self.kernel.calc_array(&self.kernel_lookup);
        let (sum, _) = sum_all(&self.kernel_lookup);
        self.kernel_lookup = div(&self.kernel_lookup, &sum, false);
        // convolution will always be equal to 1
    }

    pub fn run(&mut self, channel: &Channel) {
        // apply padding
        let p = Dim4::new(&[self.radius as u64, self.radius as u64, 0, 0]);
        let dims = channel.matrix.dims().get().to_vec();
        let temp = pad(&channel.matrix, p, p, BorderType::PERIODIC);

        self.matrix_out = convolve2(&temp, &self.kernel_lookup, ConvMode::DEFAULT, ConvDomain::FREQUENCY);

        //remove padding
        let seqs = [
            Seq::new(self.radius as u32, dims[0] as u32 + self.radius as u32 - 1, 1),
            Seq::new(self.radius as u32, dims[1] as u32 + self.radius as u32 - 1, 1),
        ];
        self.matrix_out = index(&self.matrix_out, &seqs).copy();
        
        self.matrix_out = self.growth_map.calc_array(&self.matrix_out);
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Channel {
    pub matrix: Array<f32>,
    pub matrix_out: Array<f32>,
    pub weights: HashMap<usize, f32>, // layer key, weight 
}
impl Channel {
    pub fn new(matrix: Array<f32>) -> Self {
        Self { 
            matrix_out: Array::<f32>::new_empty(matrix.dims()),
            matrix, 
            weights: HashMap::new(),
        }
    }

    // things to do after layer computation
    pub fn finish(&mut self, delta: f32) {
        //self.matrix_out = div(&self.matrix_out, &(self.layer_counter as f32), false );  // change is divided by amount of layers
        self.matrix_out = mul( &self.matrix_out, &delta, false);   // incorporate delta
        self.matrix = add(&self.matrix, &self.matrix_out, false);
        self.matrix = clamp(&self.matrix, &0_f32, &1_f32, false);
    }
}




impl PartialEq for Layer {
    fn eq(&self, other: &Self) -> bool {
        self.kernel == other.kernel &&
        self.growth_map == other.growth_map &&
        self.source_key == other.source_key &&
        self.radius == other.radius
    }
}
impl fmt::Debug for Layer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut l = self.source_key.to_string();
        l += "\tK: ";
        l += &format!("{:?}", self.kernel);
        l += "G: ";
        l += &format!("{:?}", self.growth_map);
        l += "R: ";
        l += &self.radius.to_string();
        write!(f, "{}", l)
    }
}

impl PartialEq for Channel {
    fn eq(&self, other: &Self) -> bool {
        self.matrix.dims() == other.matrix.dims()
    }
}
impl fmt::Debug for Channel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut l = "".to_string();
        l += &format!("{:?}", self.matrix.dims());
        write!(f, "{}", l)
    }
}
