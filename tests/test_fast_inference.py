import unittest
import torch
from cranepilot.innovation import InnovationForecaster
from cranepilot.fast_inference import fast_prediction


class CompiledFlowTests(unittest.TestCase):
    def batch(self,dtype=torch.float64):
        return {'angular_context':torch.randn(7,125,dtype=dtype)*.2,
            'initial_q':torch.randn(7,dtype=dtype)*.1,'observed_q':torch.randn(7,dtype=dtype)*.1,
            'initial_p':torch.randn(7,dtype=dtype)*.1,'omega':torch.linspace(3.,9.,7,dtype=dtype),
            'future_u':torch.rand(7,30,dtype=dtype)}

    def test_matches_two_cayley_solves_for_nontrivial_forcing(self):
        torch.manual_seed(11)
        for driven,anchor in [(False,False),(True,False),(True,True)]:
            model=InnovationForecaster(driven,measurement_anchor=anchor).double().eval()
            with torch.no_grad():
                model.cell.coefficients[-1].weight.normal_(0,.1)
                if driven:model.force_output.weight.fill_(.15);model.force_output.bias.fill_(.4)
            b=self.batch();a=torch.randn(7,30,dtype=torch.double)*.5
            with torch.no_grad():expected,_=model(b,a)
            torch.testing.assert_close(fast_prediction(model,b,a),expected,atol=1e-12,rtol=1e-11)

    def test_float32_batch_and_individual_forecasts_agree(self):
        torch.manual_seed(3);model=InnovationForecaster(True).eval()
        b=self.batch(torch.float32);a=torch.randn(7,30)*.5
        full=fast_prediction(model,b,a)
        for i in range(7):
            single=fast_prediction(model,{k:v[i:i+1] for k,v in b.items()},a[i:i+1])
            torch.testing.assert_close(full[i:i+1],single,atol=1e-6,rtol=1e-5)

    def test_training_mode_cannot_silently_use_inference_only_flow(self):
        with self.assertRaises(ValueError):
            fast_prediction(InnovationForecaster(True),{},torch.zeros(1,30))
