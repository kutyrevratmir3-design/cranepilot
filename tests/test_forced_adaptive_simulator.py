import copy
import unittest
import numpy as np
from cranepilot.physics import LengthCalibration
from cranepilot.data import QUANTIZATION_RAD
from scripts.forced_adaptive_simulator import simulate_forced


class ForcedAdaptiveSimulatorTests(unittest.TestCase):
    def setUp(self):
        self.calibration = LengthCalibration()
        self.cfg = {'length_m': [.23, .61], 'amplitude_deg': [7.], 'phase_rad': [.4],
                    'omega_ratio': [.82, 1.18], 'damping_per_s': [.16],
                    'drive_amplitude_m': [0., .012], 'force_law': ['effective', 'geometric'],
                    'measurement': ['exact', 'noisy_quantized'],
                    'noise_seed': 8112026, 'noise_std_bins': .25}

    def test_deterministic_measurement_pairing_and_history_only_contract(self):
        meta, packet, acceleration, target = simulate_forced(self.cfg, self.calibration)
        meta2, packet2, acceleration2, target2 = simulate_forced(self.cfg, self.calibration)
        self.assertTrue(meta.equals(meta2))
        self.assertEqual(set(packet), {'theta', 'x', 'reference', 'length_m', 'future_reference'})
        self.assertEqual(packet['theta'].shape, (32, 41)); self.assertEqual(target.shape, (32, 30))
        for key in packet: np.testing.assert_array_equal(packet[key], packet2[key])
        np.testing.assert_array_equal(acceleration, acceleration2)
        np.testing.assert_array_equal(target, target2)
        np.testing.assert_array_equal(target[:16], target[16:])
        np.testing.assert_array_equal(acceleration[:16], acceleration[16:])
        np.testing.assert_array_equal(packet['x'][:16], packet['x'][16:])
        np.testing.assert_array_equal(packet['reference'], 0)
        np.testing.assert_array_equal(packet['future_reference'], 0)
        self.assertGreater(np.max(abs(packet['theta'][:16]-packet['theta'][16:])), 0)
        np.testing.assert_allclose(packet['theta'][16:]/QUANTIZATION_RAD,
                                   np.round(packet['theta'][16:]/QUANTIZATION_RAD), atol=1e-12)
        zero = meta.drive_amplitude_m.to_numpy() == 0
        for mode in self.cfg['measurement']:
            effective = np.flatnonzero(zero & (meta.force_law == 'effective') & (meta.measurement == mode))
            geometric = np.flatnonzero(zero & (meta.force_law == 'geometric') & (meta.measurement == mode))
            np.testing.assert_array_equal(packet['theta'][effective], packet['theta'][geometric])
            np.testing.assert_array_equal(target[effective], target[geometric])

    def test_free_small_angle_solution_has_correct_time_origin_and_order(self):
        cfg = dict(self.cfg, length_m=[.61], amplitude_deg=[1e-5], omega_ratio=[1.18],
                   damping_per_s=[0.], drive_amplitude_m=[0.], force_law=['effective'], measurement=['exact'])
        _, packet, acceleration, target = simulate_forced(cfg, self.calibration)
        w = self.calibration.omega(np.array([.61]))[0]*1.18
        a = np.deg2rad(1e-5)
        np.testing.assert_allclose(packet['theta'][0], a*np.cos(.4+w*(.4-np.arange(41)*.01)), atol=2e-15, rtol=0)
        np.testing.assert_allclose(target[0], a*np.cos(.4+w*(.4+np.arange(1,31)*.05)), atol=2e-15, rtol=0)
        np.testing.assert_array_equal(acceleration, 0)

    def test_forced_small_angle_matches_independent_exact_zoh_solution(self):
        cfg = dict(self.cfg, length_m=[.61], amplitude_deg=[0.], omega_ratio=[1.18],
                   damping_per_s=[0.], drive_amplitude_m=[1e-7], force_law=['effective'], measurement=['exact'])
        _, packet, acceleration, target = simulate_forced(cfg, self.calibration)
        w = self.calibration.omega(np.array([.61]))[0]*1.18
        drive_w = 2*np.pi*.7; phase=.3; x_amplitude=1e-7; gain=w*w/9.81
        B = gain*x_amplitude*drive_w**2/(w*w-drive_w**2)
        C = -B*np.sin(drive_w*(-.4)+phase)
        D = -B*drive_w*np.cos(drive_w*(-.4)+phase)/w
        t = -np.arange(41)*.01
        exact = B*np.sin(drive_w*t+phase)+C*np.cos(w*(t+.4))+D*np.sin(w*(t+.4))
        np.testing.assert_allclose(packet['theta'][0], exact, atol=2e-15, rtol=0)
        self.assertLess(packet['theta'][0,-2], 0)  # positive initial a drives negative sway
        np.testing.assert_allclose(packet['x'][0], x_amplitude*np.sin(drive_w*t+phase), atol=0, rtol=1e-14)
        prescribed = -x_amplitude*drive_w**2*np.sin(drive_w*np.arange(30)*.05+phase)
        np.testing.assert_allclose(acceleration[0], prescribed, atol=1e-20, rtol=1e-14)
        q = exact[0]
        v = B*drive_w*np.cos(phase)-C*w*np.sin(w*.4)+D*w*np.cos(w*.4)
        future=[]; cosine=np.cos(w*.05); sine=np.sin(w*.05)
        for acc in prescribed:
            equilibrium = -gain*acc/(w*w)
            q,v = equilibrium+cosine*(q-equilibrium)+sine*v/w, -w*sine*(q-equilibrium)+cosine*v
            future.append(q)
        np.testing.assert_allclose(target[0], future, atol=2e-15, rtol=0)

    def test_nonlinear_step_halving_and_force_law_mismatch(self):
        meta, packet, _, target = simulate_forced(self.cfg, self.calibration)
        _, finer_packet, _, finer_target = simulate_forced(self.cfg, self.calibration, dt=.0005)
        self.assertLess(np.max(abs(np.rad2deg(target-finer_target))), 1e-7)
        self.assertLess(np.max(abs(np.rad2deg(packet['theta']-finer_packet['theta']))), 1e-7)
        driven = meta.drive_amplitude_m.to_numpy() > 0
        effective=np.flatnonzero(driven & (meta.force_law == 'effective') & (meta.measurement == 'exact'))
        geometric=np.flatnonzero(driven & (meta.force_law == 'geometric') & (meta.measurement == 'exact'))
        self.assertGreater(np.max(abs(target[effective]-target[geometric])), 1e-4)

    def test_future_actuator_gain_is_hidden_from_history_and_prescription(self):
        cfg = dict(self.cfg, future_acceleration_gain=[1.,1.15])
        meta, packet, acceleration, target = simulate_forced(cfg,self.calibration)
        left = np.flatnonzero(meta.future_acceleration_gain == 1.)
        right = np.flatnonzero(meta.future_acceleration_gain == 1.15)
        for key in packet: np.testing.assert_array_equal(packet[key][left],packet[key][right])
        np.testing.assert_array_equal(acceleration[left],acceleration[right])
        np.testing.assert_array_equal(meta.noise_pair_id.to_numpy()[left],meta.noise_pair_id.to_numpy()[right])
        moving = meta.drive_amplitude_m.to_numpy()[left] > 0
        np.testing.assert_array_equal(target[left][~moving],target[right][~moving])
        self.assertGreater(np.max(abs(target[left][moving]-target[right][moving])),1e-4)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError): simulate_forced(self.cfg, self.calibration, dt=.003)
        for key,value in [('force_law',['unknown']),('omega_ratio',[0]),('drive_amplitude_m',[-.1]),('measurement',['future']),('noise_std_bins',-1)]:
            bad=copy.deepcopy(self.cfg);bad[key]=value
            with self.assertRaises(ValueError): simulate_forced(bad,self.calibration)


if __name__ == '__main__': unittest.main()
