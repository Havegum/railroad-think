use dfdx::{prelude::*, tensor::Cpu};

struct Model<E, D>
where
    D: Device<E>,
    E: Dtype,
{
    conv_1: modules::Conv2D<12, 3, 3, 1, 0, 0, 1, E, D>,
    conv_1_w: Tensor<Rank4<3, 3, 12, 12>, E, D>,
    conv_2: modules::Conv2D<3, 3, 3, 1, 0, 0, 1, E, D>,
    conv_2_w: Tensor<Rank4<3, 3, 12, 3>, E, D>,
    flatten: Flatten2D,
    conv_out: modules::Linear<147, 16, E, D>,
    relu: ReLU,
    heuristic_out: modules::Linear<7, 16, E, D>,
    out: modules::Linear<32, 1, E, D>,
}

impl<E: Dtype, D: Device<E>> TensorCollection<E, D> for Model<E, D>
where
    dfdx::nn::modules::Conv2D<12, 3, 3, 1, 0, 0, 1, E, D>: dfdx::nn::TensorCollection<E, D>,
{
    type To<E2: Dtype, D2: Device<E2>> = Model<E2, D2>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                // Define name of each field and how to access it, using ModuleField for Modules,
                // and TensorField for Tensors.
                Self::module(
                    "conv_1", //
                    |s| &s.conv_1,
                    |s| &mut s.conv_1,
                ),
                Self::module(
                    "conv_2", //
                    |s| &s.conv_2,
                    |s| &mut s.conv_2,
                ),
                Self::module(
                    "conv_out", //
                    |s| &s.conv_out,
                    |s| &mut s.conv_out,
                ),
                Self::module(
                    "heuristic_out", //
                    |s| &s.heuristic_out,
                    |s| &mut s.heuristic_out,
                ),
                Self::module(
                    "out", //
                    |s| &s.out,
                    |s| &mut s.out,
                ),
            ),
            // Define how to construct the collection given its fields in the order they are given
            // above. This conversion is done using the ModuleFields trait.
            |(conv_1, conv_2, conv_out, heuristic_out, out)| Model {
                conv_1,
                conv1_w: Default::default(),
                conv_2,
                conv2_w: Default::default(),
                conv_out,
                heuristic_out,
                out,
                flatten: Default::default(),
                relu: Default::default(),
            },
        )
    }
}

// Abandoned because I'd have to figure out how to run tape through this ...
// Next step is probably to try implementing Module for Stack.
impl<E, D> Module<(Tensor<Rank3<7, 7, 12>, E, D>, Tensor<Rank1<7>, E, D>)> for Model<E, D>
where
    D: Device<E>,
    E: Dtype,
{
    type Output = Tensor<(Const<1>,), E, D>;
    type Error = <D as HasErr>::Err;

    fn try_forward(
        &self,
        x: (Tensor<Rank3<7, 7, 12>, E, D>, Tensor<Rank1<7>, E, D>),
    ) -> Result<Self::Output, Self::Error> {
        let x_1: Tensor<Rank1<16>, E, D, _> = self.convolutional.try_forward(x.0)?;
        let x_2: Tensor<Rank1<16>, E, D, _> = self.heuristic.try_forward(x.1)?;

        let x: Tensor<Rank1<32>, E, D, _> = (x_1, x_2).try_concat_along(Axis::<0>)?;
        let x: Tensor<Rank1<1>, E, D, _> = self.shared.try_forward(x)?;

        Ok(x)
    }
}

// impl
//     ModuleMut<(
//         Tensor<Rank3<7, 7, 12>, f32, Cpu>,
//         Tensor<Rank1<7>, f32, Cpu>,
//     )> for Model<Cpu, f32>
// {
//     type Output = Tensor<Rank1<1>, f32, Cpu>;
//     type Error = <Cpu as HasErr>::Err;

//     fn try_forward_mut(&self, x: Input) -> Self::Output {
//         let x_1: Tensor<Rank1<16>, f32, _, _> = self.convolutional.forward(x.0);
//         let x_2: Tensor<Rank1<16>, f32, _, _> = self.heuristic.forward(x.1);

//         let x: Tensor<Rank1<32>, _, _, _> = (x_1, x_2).try_concat_along(Axis::<0>).unwrap();
//         let x: Tensor<Rank1<1>, _, _, _> = self.shared.forward(x);

//         x
//     }
// }
