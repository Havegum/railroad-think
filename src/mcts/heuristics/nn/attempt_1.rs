use dfdx::{prelude::*, tensor::Cpu};

struct Model<E, D>
where
    D: Device<E>,
    E: Dtype,
{
    convolutional: (
        modules::Conv2D<12, 3, 3, 1, 0, 0, 1, E, D>,
        ReLU,
        modules::Conv2D<3, 3, 3, 1, 0, 0, 1, E, D>,
        Flatten2D,
        modules::Linear<147, 16, E, D>,
        ReLU,
    ),
    heuristic: (modules::Linear<7, 16, E, D>, ReLU),
    shared: modules::Linear<32, 1, E, D>,
}

impl<E: Dtype, D: Device<E>> TensorCollection<E, D> for Model<E, D> {
    type To<E2: Dtype, D2: Device<E2>> = Mlp<E2, D2>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                // Define name of each field and how to access it, using ModuleField for Modules,
                // and TensorField for Tensors.
                Self::module(
                    "convolutional", //
                    |s| &s.convolutional,
                    |s| &mut s.convolutional,
                ),
                Self::module(
                    "heuristic", //
                    |s| &s.heuristic,
                    |s| &mut s.heuristic,
                ),
                Self::module(
                    "shared", //
                    |s| &s.shared,
                    |s| &mut s.shared,
                ),
            ),
            // Define how to construct the collection given its fields in the order they are given
            // above. This conversion is done using the ModuleFields trait.
            |(convolutional, heuristic, shared)| Model {
                convolutional,
                heuristic,
                shared,
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
