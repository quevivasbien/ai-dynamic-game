pub trait CSF {
    fn q_i(&self, i: usize, p: &Vec<f64>) -> f64;
    fn q(&self, p: &Vec<f64>) -> Vec<f64> {
        (0..p.len()).map(|i| self.q_i(i, p)).collect()
    }
}

pub struct DefaultCSF;

impl CSF for DefaultCSF {
    fn q_i(&self, i: usize, p: &Vec<f64>) -> f64 {
        let sum_p: f64 = p.iter().sum();
        if sum_p == 0.0 {
            0.0
        } else {
            p[i] / sum_p
        }
    }

    fn q(&self, p: &Vec<f64>) -> Vec<f64> {
        let sum_p: f64 = p.iter().sum();
        if sum_p == 0.0 {
            vec![0.0; p.len()]
        } else {
            p.iter().map(|x| x / sum_p).collect()
        }
    }
}
