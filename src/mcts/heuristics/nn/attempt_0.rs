use dfdx::{prelude::*, tensor::Cpu};

// type ConvModel = (
//     Conv2D<7, 3, 3>,
//     Tanh,
//     Conv2D<3, 3, 3>,
//     Flatten2D,
//     Linear<147, 16>,
//     ReLU,
// );

// type HeuristicsModel = (Linear<7, 16>, ReLU, Linear<16, 16>, ReLU);

type F = f32;
type DS = Cpu;

mod builder {
    use super::*;

    pub struct ModelTwo;

    impl BuildOnDevice<DS, F> for ModelTwo {
        type Built = super::ModelTwo<F, DS>;
    }
}

pub struct ModelTwo<F: Dtype, DS: Device<F>> {
    convolutional: (
        modules::Conv2D<7, 3, 3, 1, 0, 0, 1, F, DS>,
        ReLU,
        modules::Conv2D<3, 3, 3, 1, 0, 0, 1, F, DS>,
        Flatten2D,
        modules::Linear<147, 16, F, DS>,
        ReLU,
    ),
    heuristic: (modules::Linear<7, 16, F, DS>, ReLU),
    shared: modules::Linear<32, 1, F, DS>,
}

impl TensorCollection<F, DS> for ModelTwo<F, DS> {
    type To<F2: Dtype, DS2: Device<F2>> = ModelTwo<F2, DS2>;

    fn iter_tensors<V: ModuleVisitor<Self, F, DS>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        todo!();
        // visitor.visit_fields(
        //     Self::module(
        //         "0", //
        //         |s| &s.shared,
        //         |s| &mut s.shared,
        //     ),
        //     builder::ModelTwo,
        // )
    }
}

// also impl Module as well as batched Module and ModuleMut
impl<T: Tape<F, DS>>
    ModuleMut<(
        Tensor<Rank3<7, 7, 17>, F, DS, T>,
        Tensor<Rank1<7>, F, DS, T>,
    )> for ModelTwo<F, DS>
{
    type Output = Tensor<Rank1<1>, F, DS, T>;
    type Error = <DS as HasErr>::Err;

    fn try_forward_mut(
        &mut self,
        x: (Tensor<Rank3<7, 7, 17>, F, DS>, Tensor<Rank1<7>, F, DS>),
    ) -> Result<Self::Output, Self::Error> {
        let a = x.split_tape();

        let conv_in: Tensor<Rank3<7, 7, 12>, _, _> = x.0;
        let heuristics_in: Tensor<Rank1<7>, F, DS> = x.1;

        let x1: Tensor<Rank1<16>, _, _> = self.convolutional.try_forward_mut(conv_in)?;
        let x2: Tensor<Rank1<16>, _, _> = self.heuristic.try_forward_mut(heuristics_in)?;
        let x: Tensor<Rank1<32>, _, _> = (x1, x2).try_concat_along(Axis::<0>)?;
        let x = self.shared.try_forward_mut(x)?;
        Ok(x)
    }
}

impl Module<Tensor<Rank1<512>, F, DS>> for ModelTwo<F, DS> {
    type Output = Tensor<Rank1<512>, F, DS>;
    type Error = <DS as HasErr>::Err;

    fn try_forward(&self, x: Tensor<Rank1<512>, F, DS>) -> Result<Self::Output, Self::Error> {
        Ok(x)
    }
}

// type Model = (ConvModel, TryConcat<HeuristicsModel>, Linear<32, 1>);
// type Output = (Linear<32, 1>, ReLU);
// struct Concat {
//     conv_model: ConvModel,
//     heuristics_model: HeuristicsModel,
//     output: Output,
// }

//  impl trait BuildOnDevice:
// impl BuildOnDevice<Cpu, f32> for Concat {
//     type Built = BuildModule<Cpu, f32>;

//     fn build_on_device(device: &Cpu) -> Self {
//         let conv_model = ConvModel::build_on_device(device);
//         let heuristics_model = HeuristicsModel::build_on_device(device);
//         let output = Output::build_on_device(device);
//         Self {
//             conv_model,
//             heuristics_model,
//             output,
//         }
//     }
// }

// impl Module<(Tensor3D<7, 7, 12>, Tensor1D<7>)> for Concat {
//     type Output = Rank1<1>;
//     type Error = Option<usize>;

//     fn try_forward(
//         &self,
//         (c, h): (Tensor3D<7, 7, 12>, Tensor1D<7>),
//     ) -> Result<Self::Output, Self::Error> {
//         let c = self.conv_model.forward(c);
//         let h = self.heuristics_model.forward(h);
//         let x = (c, h).concat_along(Axis::<0>);
//         let x = self.output.forward(x);
//         Ok(x)
//     }
// }

// #[derive(Debug, Default, Clone)]
// pub struct AddInto<T>(pub T);

// impl<T: BuildOnDevice<D, E>, D: Device<E>, E: Dtype> BuildOnDevice<D, E> for AddInto<T> {
//     type Built = AddInto<T::Built>;
// }

// impl<E: Dtype, D: Device<E>, T: TensorCollection<E, D>> TensorCollection<E, D> for AddInto<T> {
//     type To<E2: Dtype, D2: Device<E2>> = AddInto<T::To<E2, D2>>;

//     fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
//         visitor: &mut V,
//     ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
//         visitor.visit_fields(
//             Self::module(
//                 "0", //
//                 |s| &s.0,
//                 |s| &mut s.0,
//             ),
//             AddInto,
//         )
//     }
// }

// struct SomeModule;

// impl Module<(TupleType1, TupleType2)> for SomeModule {
//     type Output = ModuleOutput;
//     fn try_forward(&self, (t1, t2): (TupleType1, TupleType2)) -> Result<Self::Output, Error> {
//         // logic here
//         Ok(ModuleOutput)
//     }
// }
