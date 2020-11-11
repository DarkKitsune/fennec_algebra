use crate::{init_array, Dot, RandNorm, Sqr};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

type ValueType = f64;

pub type NNetResult<T> = Result<T, NNetError>;

#[derive(Debug)]
#[repr(C)]
pub struct Layer<const WIDTH: usize, const NEXT_WIDTH: usize> {
    outputs: [ValueType; WIDTH],
    weights: [[ValueType; WIDTH]; NEXT_WIDTH],
    bias: ValueType,
}

impl<const WIDTH: usize, const NEXT_WIDTH: usize> Layer<WIDTH, NEXT_WIDTH> {
    fn new(seed: &mut u64) -> Self {
        Self {
            outputs: init_array!([ValueType; WIDTH], |_| ValueType::default()),
            weights: init_array!([[ValueType; WIDTH]; NEXT_WIDTH], mut |_| init_array!([ValueType; WIDTH], mut |_| ValueType::rand_next(seed))),
            bias: ValueType::rand_next(seed),
        }
    }

    fn calculate_outputs<const PREVIOUS_WIDTH: usize, const PREVIOUS_NEXT_WIDTH: usize>(
        &mut self,
        previous: &Layer<PREVIOUS_WIDTH, PREVIOUS_NEXT_WIDTH>,
    ) {
        let mut outputs = previous
            .weights
            .iter()
            .take(WIDTH)
            .map(|weights| {
                fast_sigmoid(
                    previous
                        .outputs
                        .iter()
                        .enumerate()
                        .map(|(idx, prev_output)| (*prev_output, weights[idx]))
                        .dot()
                        + self.bias,
                )
            })
            .collect::<Vec<ValueType>>();
        for (idx, output) in outputs.drain(..).enumerate() {
            self.outputs[idx] = output;
        }
    }

    fn cost(&self, targets: impl Iterator<Item = ValueType>) -> ValueType {
        let sum: ValueType = self
            .outputs
            .iter()
            .zip(targets)
            .map(|(output, target)| (target - output).sqr())
            .sum();
        (1.0 / WIDTH as ValueType) * sum
    }

    fn backpropagate<
        const PREVIOUS_WIDTH: usize,
        const PREVIOUS_WEIGHT_COUNT: usize,
        const OUTPUT_WIDTH: usize,
    >(
        &mut self,
        targets: &[ValueType; OUTPUT_WIDTH],
        previous: &mut Layer<PREVIOUS_WIDTH, PREVIOUS_WEIGHT_COUNT>,
        learning_rate: ValueType,
    ) {
        let mut weight_bias_deltas = self
            .outputs
            .iter()
            .zip(targets)
            .map(|(&output, target)| {
                let error = output - target;
                let dcost_dpred = error;
                let dpred_dz = fast_sigmoid_derivative(output);
                let z_delta = dcost_dpred * dpred_dz;
                let inputs = previous.outputs.iter().map(|input| [*input]);
                let mut weight_deltas = inputs.map(|input| -learning_rate * input[0] * z_delta);
                (
                    init_array!([ValueType; PREVIOUS_WIDTH], mut |_| weight_deltas.next().unwrap()),
                    -learning_rate * z_delta,
                )
            })
            .collect::<Vec<([ValueType; PREVIOUS_WIDTH], ValueType)>>();
        for (idx, (weight_delta, bias_delta)) in weight_bias_deltas.drain(..).enumerate() {
            for (idx, weight) in previous.weights[idx].iter_mut().enumerate() {
                *weight += weight_delta[idx];
            }
            self.bias += bias_delta;
        }
    }

    fn load(&mut self, reader: &mut BufReader<File>) -> NNetResult<()> {
        let mut outputs_buffer = (0..WIDTH * std::mem::size_of::<ValueType>())
            .map(|_| 0u8)
            .collect::<Vec<u8>>();
        let mut weights_buffer = (0..NEXT_WIDTH * WIDTH * std::mem::size_of::<ValueType>())
            .map(|_| 0u8)
            .collect::<Vec<u8>>();
        let mut bias_buffer = (0..std::mem::size_of::<ValueType>())
            .map(|_| 0u8)
            .collect::<Vec<u8>>();
        reader
            .read(&mut outputs_buffer)
            .map_err(|_| NNetError::CouldNotReadBuffer)?;
        reader
            .read(&mut weights_buffer)
            .map_err(|_| NNetError::CouldNotReadBuffer)?;
        reader
            .read(&mut bias_buffer)
            .map_err(|_| NNetError::CouldNotReadBuffer)?;
        for idx in 0..WIDTH {
            unsafe {
                self.outputs[idx] = *(outputs_buffer.as_ptr().add(idx) as *const ValueType);
            }
        }
        for idx in 0..NEXT_WIDTH {
            for idx2 in 0..WIDTH {
                unsafe {
                    self.weights[idx][idx2] =
                        *(weights_buffer.as_ptr().add(idx) as *const ValueType);
                }
            }
        }
        unsafe {
            self.bias = *(bias_buffer.as_ptr() as *const ValueType);
        }
        Ok(())
    }

    fn save(&self, writer: &mut BufWriter<File>) -> NNetResult<()> {
        let mut buffer = (0..(WIDTH + (NEXT_WIDTH * WIDTH) + 1) * std::mem::size_of::<ValueType>())
            .map(|_| 0u8)
            .collect::<Vec<u8>>();
        let buffer_start = buffer.as_mut_ptr() as *mut ValueType;
        let mut buffer_cursor = 0;
        for &value in self.outputs.iter() {
            unsafe {
                *buffer_start.add(buffer_cursor) = value;
                buffer_cursor += 1;
            }
        }
        for weights in self.weights.iter() {
            for &value in weights.iter() {
                unsafe {
                    *buffer_start.add(buffer_cursor) = value;
                    buffer_cursor += 1;
                }
            }
        }
        unsafe {
            *buffer_start.add(buffer_cursor) = self.bias;
        }
        writer
            .write(&buffer)
            .map_err(|_| NNetError::CouldNotWriteBuffer)?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct NNet<
    const INPUT_WIDTH: usize,
    const OUTPUT_WIDTH: usize,
    const LAYER_WIDTH: usize,
    const LAYER_COUNT: usize,
> {
    learning_rate: ValueType,
    input: Option<Layer<INPUT_WIDTH, LAYER_WIDTH>>,
    output: Option<Layer<OUTPUT_WIDTH, 0>>,
    layers: [Option<Layer<LAYER_WIDTH, LAYER_WIDTH>>; LAYER_COUNT],
}

impl<
        const INPUT_WIDTH: usize,
        const OUTPUT_WIDTH: usize,
        const LAYER_WIDTH: usize,
        const LAYER_COUNT: usize,
    > NNet<INPUT_WIDTH, OUTPUT_WIDTH, LAYER_WIDTH, LAYER_COUNT>
{
    pub fn new(seed: &mut u64, learning_rate: ValueType) -> NNetResult<Self> {
        if OUTPUT_WIDTH > LAYER_WIDTH {
            return Err(NNetError::OutputLargerThanHiddenLayer);
        }

        let layers = init_array!([Option<Layer<LAYER_WIDTH, LAYER_WIDTH>>; LAYER_COUNT], mut |_| Some(Layer::new(seed)));

        Ok(Self {
            learning_rate,
            input: Some(Layer::new(seed)),
            output: Some(Layer::new(seed)),
            layers,
        })
    }

    pub fn forward(
        &mut self,
        inputs: &[ValueType; INPUT_WIDTH],
    ) -> NNetResult<[ValueType; OUTPUT_WIDTH]> {
        for (idx, &input) in inputs.iter().enumerate() {
            self.input.as_mut().unwrap().outputs[idx] = input;
        }

        let mut swapped_layer = None;
        std::mem::swap(&mut swapped_layer, &mut self.input);
        self.layers[0]
            .as_mut()
            .unwrap()
            .calculate_outputs(swapped_layer.as_ref().unwrap());
        std::mem::swap(&mut swapped_layer, &mut self.input);

        for idx in 1..LAYER_COUNT {
            let mut swapped_layer = None;
            std::mem::swap(&mut swapped_layer, &mut self.layers[idx - 1]);
            self.layers[idx]
                .as_mut()
                .expect(&format!("Hidden layer {} is none", idx))
                .calculate_outputs(
                    swapped_layer
                        .as_ref()
                        .expect(&format!("Hidden layer {} is none", idx - 1)),
                );
            std::mem::swap(&mut swapped_layer, &mut self.layers[idx - 1]);
        }

        let mut swapped_layer = None;
        std::mem::swap(&mut swapped_layer, &mut self.layers[LAYER_COUNT - 1]);
        self.output
            .as_mut()
            .unwrap()
            .calculate_outputs(swapped_layer.as_ref().unwrap());
        std::mem::swap(&mut swapped_layer, &mut self.layers[LAYER_COUNT - 1]);

        Ok(self.output.as_ref().unwrap().outputs)
    }

    pub fn backward(&mut self, targets: &[ValueType; OUTPUT_WIDTH]) {
        let mut swapped_layer = None;
        std::mem::swap(&mut swapped_layer, &mut self.input);
        self.layers[0].as_mut().unwrap().backpropagate(
            targets,
            swapped_layer.as_mut().unwrap(),
            self.learning_rate,
        );
        std::mem::swap(&mut swapped_layer, &mut self.input);

        for idx in 1..LAYER_COUNT {
            let mut swapped_layer = None;
            std::mem::swap(&mut swapped_layer, &mut self.layers[idx - 1]);
            self.layers[idx].as_mut().unwrap().backpropagate(
                targets,
                swapped_layer.as_mut().unwrap(),
                self.learning_rate,
            );
            std::mem::swap(&mut swapped_layer, &mut self.layers[idx - 1]);
        }

        let mut swapped_layer = None;
        std::mem::swap(&mut swapped_layer, &mut self.layers[LAYER_COUNT - 1]);
        self.output.as_mut().unwrap().backpropagate(
            targets,
            swapped_layer.as_mut().unwrap(),
            self.learning_rate,
        );
        std::mem::swap(&mut swapped_layer, &mut self.layers[LAYER_COUNT - 1]);
    }

    pub fn cost(&self, targets: impl Iterator<Item = ValueType>) -> ValueType {
        self.output.as_ref().unwrap().cost(targets)
    }

    pub fn save_layers(&self, writer: &mut BufWriter<File>) -> NNetResult<()> {
        self.input.as_ref().unwrap().save(writer)?;
        self.output.as_ref().unwrap().save(writer)?;
        for layer in self.layers.iter() {
            layer.as_ref().unwrap().save(writer)?;
        }
        Ok(())
    }

    pub fn load_layers(&mut self, reader: &mut BufReader<File>) -> NNetResult<()> {
        self.input.as_mut().unwrap().load(reader)?;
        self.output.as_mut().unwrap().load(reader)?;
        for layer in self.layers.iter_mut() {
            layer.as_mut().unwrap().load(reader)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub enum NNetError {
    OutputLargerThanHiddenLayer,
    CouldNotWriteBuffer,
    CouldNotReadBuffer,
}

impl NNetError {
    pub fn message(&self) -> &'static str {
        match self {
            NNetError::OutputLargerThanHiddenLayer => {
                "Output layer cannot be larger than hidden layers"
            }
            NNetError::CouldNotWriteBuffer => "Could not write to buffer when saving",
            NNetError::CouldNotReadBuffer => "Could not read from buffer when loading",
        }
    }
}

fn fast_sigmoid(x: ValueType) -> ValueType {
    (x * 0.5) / (x.abs() + 1.0) + 0.5
}

fn fast_sigmoid_derivative(x: ValueType) -> ValueType {
    0.5 / (x.abs() + 1.0).sqr()
}

impl std::fmt::Display for NNetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message())
    }
}

impl Into<String> for NNetError {
    fn into(self) -> String {
        String::from(self.message())
    }
}

impl AsRef<str> for NNetError {
    fn as_ref(&self) -> &str {
        self.message()
    }
}
