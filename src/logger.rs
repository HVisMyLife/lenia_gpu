use std::{collections::HashMap, fs::{self, File}, io::{Read, Write}, vec};
use arrayfire::{Array, Dim4};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use crate::{Channel, Function, Layer, Lenia};


#[derive(Clone, Serialize, Deserialize)]
pub struct PackageLenia {
    pub lenia: DataLenia,
    pub layers: Vec<(usize, DataLayer)>,
    pub channels: Vec<(usize, DataChannel)>
}
impl PackageLenia {
    pub fn empty() -> Self {
        Self { lenia: DataLenia{ delta: 0.1}, layers: vec![], channels: vec![] }
    }
    pub fn from_lenia(lenia: &Lenia) -> Self {
        let l = DataLenia::new(lenia);
        let layers: Vec<(usize, DataLayer)> = 
            lenia.layers.iter().map(|(k,l)| (*k, DataLayer::new(l)) ).collect();
        let channels: Vec<(usize, DataChannel)> = 
            lenia.channels.iter().map(|(k,ch)| (*k, DataChannel::new(ch)) ).collect();
        Self { 
            lenia: l,
            layers, 
            channels 
        }
    }
    pub fn update_lenia(package: &Self, lenia: &mut Lenia) {
        lenia.delta = package.lenia.delta;
        package.layers.iter().for_each(|(k,l)|{
            let test = lenia.layers.get(k).is_some();
            let layer = if test {lenia.layers.get_mut(k).unwrap()} else {
                let copy = lenia.layers.iter().next().unwrap().1.clone();
                lenia.layers.insert(*k, copy).unwrap();
                lenia.layers.get_mut(k).unwrap()
            };

            layer.source_key = l.source_key;
            layer.kernel = l.kernel.clone();
            layer.growth_map = l.growth_map.clone();
            layer.radius = l.radius;
            layer.generate_kernel_lookup();
        });
        package.channels.iter().for_each(|(k,ch)|{
            let test = lenia.channels.get(k).is_some();
            let channel = if test {lenia.channels.get_mut(k).unwrap()} else {
                let copy = lenia.channels.iter().next().unwrap().1.clone();
                lenia.channels.insert(*k, copy).unwrap();
                lenia.channels.get_mut(k).unwrap()
            };
            ch.keys.iter().zip(ch.floats.iter()).for_each(|(dk,f)|{
                *channel.weights.get_mut(dk).unwrap() = *f;
            });
        });
    }
}


#[derive(Clone, Serialize, Deserialize)]
pub struct DataLayer {
    pub source_key: usize,
    pub kernel: Function,
    pub growth_map: Function,
    pub radius: usize,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct DataMatrix {}
#[derive(Clone, Serialize, Deserialize)]
pub struct DataChannel {
    pub keys: Vec<usize>, // layer key, weight 
    pub floats: Vec<f32>
}

#[derive(Clone, Serialize, Deserialize)]
pub struct DataLenia {
    pub delta: f32,
}

impl DataLenia {
    fn new(lenia: &Lenia) -> Self {
        Self {delta: lenia.delta}
    }

    pub fn save(key: usize, lenia: &Lenia) {
        let mut path = "data/".to_string();
        path += &key.to_string();
        path += "/";
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(path.clone() + "matrix").unwrap();
        fs::create_dir_all(path.clone() + "channel").unwrap();
        fs::create_dir_all(path.clone() + "layer").unwrap();

        let data_lenia = DataLenia {delta: lenia.delta };
        let toml = toml::to_string(&data_lenia).unwrap();
        let mut file = File::create(path + "lenia.toml").unwrap();
        file.write(toml.as_bytes()).unwrap();

        lenia.layers.iter().sorted_by(|a,b| Ord::cmp(a.0, b.0) ).for_each(|(k,l)|{
            DataLayer::save(key, *k, &l);
        });
        
        lenia.channels.iter().sorted_by(|a,b| Ord::cmp(a.0, b.0) ).for_each(|(k,c)|{
            DataMatrix::save(key, *k, &c);
            DataChannel::save(key, *k, &c);
        });
    }
    pub fn load(key: usize) -> Lenia {
        let mut path = "data/".to_string();
        path += &key.to_string();
        path += "/";
        
        let mut toml = String::new();
        let mut file = File::open(path.clone() + "lenia.toml").unwrap();
        file.read_to_string(&mut toml).unwrap();
        let decoded: Self = toml::from_str(&toml).unwrap();
        let mut lenia = Lenia::new(decoded.delta, HashMap::new(), HashMap::new());
        
        
        let dir = fs::read_dir(path.clone() + "layer/" ).unwrap();
        let entries = dir.map(|res| res.unwrap().file_name().into_string().unwrap() ).collect::<Vec<_>>();
        entries.iter().for_each(|e|{
            let layer = DataLayer::load(key, e.replace(".toml", "").parse::<usize>().unwrap());
            lenia.layers.insert(e.replace(".toml", "").parse::<usize>().unwrap(), layer);
        });
        
        let dir = fs::read_dir(path.clone() + "matrix/" ).unwrap();
        let entries = dir.map(|res| res.unwrap().file_name().into_string().unwrap() ).collect::<Vec<_>>();
        entries.iter().for_each(|e|{
            let channel = DataMatrix::load(key, e.replace(".bin", "").parse::<usize>().unwrap());
            lenia.channels.insert(e.replace(".bin", "").parse::<usize>().unwrap(), channel);
        });
        let dir = fs::read_dir(path.clone() + "channel/" ).unwrap();
        let entries = dir.map(|res| res.unwrap().file_name().into_string().unwrap() ).collect::<Vec<_>>();
        entries.iter().for_each(|e|{
            let channel = DataChannel::load(key, e.replace(".toml", "").parse::<usize>().unwrap());
            lenia.channels.get_mut( &e.replace(".toml", "").parse::<usize>().unwrap() ).unwrap().weights = channel.weights;
        });

        lenia
    }
}


impl DataChannel {
    fn new(channel: &Channel) -> Self {
        let mut ch = Self {
            keys: vec![],
            floats: vec![]
        };
        channel.weights.iter().for_each(|(k,w)|{
            ch.keys.push(*k);
            ch.floats.push(*w);
        });
        ch
    }
    fn save(lenia_key: usize, key: usize, channel: &Channel) {
        let mut ch = Self {
            keys: vec![],
            floats: vec![]
        };

        let mut path = "data/".to_string();
        path += &lenia_key.to_string();
        path += "/channel/";
        path += &key.to_string();
        path += ".toml";
        
        channel.weights.iter().for_each(|(k,w)|{
            ch.keys.push(*k);
            ch.floats.push(*w);
        });
        let toml = toml::to_string(&ch).unwrap();
        let mut file = File::create(path).unwrap();
        file.write(toml.as_bytes()).unwrap();
    }
    fn load(lenia_key: usize, key: usize) -> Channel {
        let mut path = "data/".to_string();
        path += &lenia_key.to_string();
        path += "/channel/";
        path += &key.to_string();
        path += ".toml";
        
        let mut toml = String::new();
        let mut file = File::open(path).unwrap();
        
        file.read_to_string(&mut toml).unwrap();
        let decoded: Self = toml::from_str(&toml).unwrap();
        
        let mut weights: HashMap<usize, f32> = HashMap::new(); 
        decoded.keys.iter().zip(decoded.floats.iter()).for_each(|(k, w)|{
            weights.insert(*k, *w);
        });

        Channel { 
            matrix: Array::new_empty(Dim4::new(&[1,1,1,1])), 
            matrix_out: Array::new_empty(Dim4::new(&[1,1,1,1])), 
            weights 
        }
    }
}

impl DataMatrix {
    fn save(lenia_key: usize, key: usize, channel: &Channel) {
        let mut path = "data/".to_string();
        path += &lenia_key.to_string();
        path += "/matrix/";
        path += &key.to_string();
        path += ".bin";
        let encoded: Vec<u8> = bincode::serialize(&channel.matrix).unwrap();
        let mut file = File::create(path).unwrap();
        file.write(&encoded).unwrap();
    }
    fn load(lenia_key: usize, key: usize) -> Channel {
        let mut path = "data/".to_string();
        path += &lenia_key.to_string();
        path += "/matrix/";
        path += &key.to_string();
        path += ".bin";
        let mut file = File::open(path).unwrap();
        let mut buffer = vec![];
        file.read_to_end(&mut buffer).unwrap();
        let matrix: Array<f32> = bincode::deserialize(&buffer).unwrap();

        Channel { matrix: matrix.copy(), matrix_out: matrix, weights: HashMap::new() }
    }
}

impl DataLayer {
    fn new(layer: &Layer) -> Self {
        Self {
            source_key: layer.source_key,
            kernel: layer.kernel.clone(),
            growth_map: layer.growth_map.clone(),
            radius: layer.radius
        }
    }
    fn save(lenia_key: usize, key: usize, layer: &Layer) {
        let tl = Self {
            source_key: layer.source_key,
            kernel: layer.kernel.clone(),
            growth_map: layer.growth_map.clone(),
            radius: layer.radius
        };

        let mut path = "data/".to_string();
        path += &lenia_key.to_string();
        path += "/layer/";
        path += &key.to_string();
        path += ".toml";

        let toml = toml::to_string(&tl).unwrap();
        let mut file = File::create(path).unwrap();
        file.write(toml.as_bytes()).unwrap();
    }
    fn load(lenia_key: usize, key: usize) -> Layer {
        let mut path = "data/".to_string();
        path += &lenia_key.to_string();
        path += "/layer/";
        path += &key.to_string();
        path += ".toml";
        
        let mut toml = String::new();
        let mut file = File::open(path).unwrap();

        file.read_to_string(&mut toml).unwrap();
        let decoded: DataLayer = toml::from_str(&toml).unwrap();

        let mut layer = Layer { 
            kernel: decoded.kernel,
            kernel_lookup: Array::new_empty(Dim4::new(&[1,1,1,1])), 
            growth_map: decoded.growth_map, 
            source_key: decoded.source_key, 
            matrix_out: Array::new_empty(Dim4::new(&[1,1,1,1])), 
            radius: decoded.radius 
        };
        layer.generate_kernel_lookup();
        layer
    }
}

