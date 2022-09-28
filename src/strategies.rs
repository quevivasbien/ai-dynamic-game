// represents action for a single player in a single time period
#[derive(Clone, Copy, Debug)]
pub struct Action {
    xs: f64,
    xp: f64,
}

// represents actions for n players in a single time period
#[derive(Clone, Debug)]
pub struct ActionSet {
    pub n: usize,
    pub xs: Vec<f64>,
    pub xp: Vec<f64>,
}

impl ActionSet {
    pub fn new(xs: Vec<f64>, xp: Vec<f64>) -> Self {
        let n = xs.len();
        assert_eq!(n, xp.len(), "xs and xp must have equal length");
        ActionSet { n, xs, xp }
    }

    pub fn from_actions(actions: Vec<Action>) -> Self {
        let (xs, xp) = actions.iter().map(|Action { xs, xp }| (xs, xp)).unzip();
        ActionSet::new(xs, xp)
    }

    pub fn replace_action(&mut self, i: usize, action: Action) {
        self.xs[i] = action.xs;
        self.xp[i] = action.xp;
    }
}

// represents actions for a single player in t time periods
#[derive(Clone, Debug)]
pub struct Strategy {
    pub t: usize,
    pub action_sequence: Vec<Action>,
}

impl Strategy {
    pub fn new(action_sequence: Vec<Action>) -> Self{
        Strategy {
            t: action_sequence.len(),
            action_sequence
        }
    }
}

// represents actions for n players in t time periods
#[derive(Clone, Debug)]
pub struct StrategySet {
    pub t: usize,
    pub n: usize,
    pub actions_sequence: Vec<ActionSet>,
}

impl StrategySet {
    pub fn new(actions_sequence: Vec<ActionSet>) -> Self {
        let t = actions_sequence.len();
        assert!(t != 0, "Cannot construct strategy set with zero-length strategies");
        let n = actions_sequence[0].n;
        assert!(
            actions_sequence.iter().map(|x| x.n).all(|l| l == n),
            "All action sets in a strategy set must have equal n"
        );
        StrategySet { t, n, actions_sequence }
    }

    pub fn from_strategies(strategies: Vec<Strategy>) -> Self {

        assert!(strategies.len() > 0);
        let t = strategies[0].t;
        assert!(
            strategies.iter().map(|s| s.t).all(|t_| t_ == t),
            "All strategies in strategy set must have equal length"
        );
        
        // extract action sequences (n x t)
        let action_sequences: Vec<Vec<Action>> = strategies.into_iter().map(|s| s.action_sequence).collect();
        // basically transpose action sequences to be t x n
        // then can convert to vec of ActionSet
        let mut transposed_seqs: Vec<Vec<Action>> = (0..t).map(|_| Vec::<Action>::new()).collect();
        for s in action_sequences.into_iter() {
            for (t_, st_) in s.into_iter().enumerate() {
                transposed_seqs[t_].push(st_);
            }
        }
        let action_sets = transposed_seqs.into_iter().map(|a|
            ActionSet::from_actions(a)
        ).collect();
        StrategySet::new(action_sets)
    }

    pub fn replace_strategy(&mut self, i: usize, strategy: Strategy) {
        for (my_actions, action) in self.actions_sequence.iter_mut().zip(strategy.action_sequence.into_iter()) {
            my_actions.replace_action(i, action);
        }
    }
}
