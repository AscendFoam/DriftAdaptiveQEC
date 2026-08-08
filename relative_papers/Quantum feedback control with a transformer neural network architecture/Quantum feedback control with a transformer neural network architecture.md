<div align="center">

# Quantum feedback control with a transformer neural network architecture

</div>

Pranav Vaidhyanathan, $ ^{1} $ Florian Marquardt, $ ^{2,3} $ Mark T. Mitchison, $ ^{4,5,*} $ and Natalia Ares $ ^{1, \dagger} $

$ ^{1} $Department of Engineering Science, University of Oxford, Oxford OX1 3PJ, United Kingdom $ ^{2} $Max Planck Institute for the Science of Light, Staudtstr. 2, 91058 Erlangen, Germany $ ^{3} $Department of Physics, Friedrich-Alexander-Universität Erlangen-Nürnberg, 91058 Erlangen, Germany $ ^{4} $School of Physics, Trinity College Dublin, College Green, Dublin 2, D02 K8N4, Ireland $ ^{5} $Department of Physics, King's College London, Strand, London, WC2R 2LS, United Kingdom

Attention-based neural networks such as transformers have revolutionized various fields such as natural language processing, genomics, and vision. Here, we demonstrate the use of transformers for quantum feedback control through both a supervised and reinforcement learning approach. In particular, due to the transformer's ability to capture long-range temporal correlations and training efficiency, we show that it can surpass some of the limitations of previous control approaches, e.g. those based on recurrent neural networks trained using a similar approach or policy based reinforcement learning. We numerically show, for the example of state stabilization of a two-level system, that our bespoke transformer architecture can achieve near unit fidelity to a target state in a short time even in the presence of inefficient measurement and Hamiltonian perturbations that were not included in the training set as well as the control of non-Markovian systems. We also demonstrate that our transformer can perform energy minimization of non-integrable many-body quantum systems when trained for reinforcement learning tasks. Our approach can be used for quantum error correction, fast control of quantum states in the presence of colored noise, as well as real-time tuning, and characterization of quantum devices.

Introduction. Quantum technologies depend crucially on our ability to precisely control quantum systems. Measurement-based feedback is an especially powerful approach to quantum control, which lies at the heart of quantum error correction and has myriad applications in the preparation and stabilization of quantum states in the presence of noise [1][2][3]. However, unlike noisy feedback control in the classical regime, quantum feedback faces an additional obstacle: only partial information on the quantum state is available, even in principle, due to the inherently disturbing nature of quantum measurements. In general, therefore, optimizing control fields requires in addition to the measurement record an estimate of the quantum state to be explicitly computed from some model of the dynamics [4], adding overhead (e.g., extra memory or time costs) to the feedback loop.

Machine learning algorithms offer a promising route to solve this problem. Appropriately trained neural networks can provide a compact representation of the most important correlations between data, allowing for significantly more efficient feedback protocols, in principle. Recent work in this direction has demonstrated the power of both model-free [5-8] and model-based [9,10] reinforcement learning for quantum feedback control. Although the former approach is a "black box" that is flexible enough to be applied to a range of scenarios, the latter can exploit the physics of the system to improve efficiency.

However, approaches using recurrent neural networks do not scale well with long-range dependencies, such as an extensive measurement record. They also suffer from the problem of vanishing gradients [11]. This is due to the assumption that the hidden state within each recurrent unit encodes dependencies from the previous state. This

adds an inherent Markovian inductive bias that is unsuitable for processes with memory [12]. Clearly, this poses a challenge for feedback control of non-Markovian open system dynamics [13], which arises naturally in many platforms. Yet, even for Markovian open quantum systems (i.e., those described by a Lindblad equation), the measurement record is a non-Markovian stochastic process [14] because measurement backaction causes the future evolution of the state to depend unavoidably on past measurement outcomes.

In recent years, transformers and attention-based models that were originally used to model natural language [15], have emerged as extremely versatile tools for various fields, ranging from genomics to robotics [16]. Due to the attention mechanism that encodes correlation between all aspects of the given input sequence, they have far outclassed recurrent neural networks (RNNs) and long-short term memory type RNNs (LSTMs) in various tasks. Recently, transformer-based approaches have also demonstrated their ability not only to adapt to model-based and model-free reinforcement learning tasks [17][18] but also to perform on par with state-of-the-art approaches for these tasks. As highlighted by Chen et al. these causally masked transformers simply output optimal actions and eliminate the requirement to fit value functions or calculate policy gradients[18].

In this work, we aim to utilize the "unreasonable" effectiveness of the attention mechanism in order to perform closed-loop feedback control for continuously measured open quantum systems that also undergo evolution due to measurement back-action [4,19]. We demonstrate that an attention-based approach to quantum feedback control, with neural networks trained using both supervised and reinforcement learning, offers a robust and scal-

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F2026080810235104a236c681074741%2Fcrop_1_1786155859471.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=KxAIc4MMPABHidYBeYSdpWBzYR8%3D&Expires=1786760659' alt='OCR图片'/></div>

<div align="center">

FIG.1. Problem and Architecture Overview: a, The two level system (TLS) interacting with a bath in a non-Markovian manner that is embedded with a reaction coordinate (RC) which interacts in a Markovian manner. The measurement record obtained continuously is used to predict optimal values of the control parameters by the transformer. b, The transformer's structure consists of an encoder and decoder architecture. During training, the encoder takes the initial state and the measurement record as input (green dotted boxes). The decoder takes the encoder output as part of the cross attention layer and the optimal parameters (blue dotted boxes), to autoregressively predict the optimal parameters for the next time steps. However, during inference, only the initial state and measurement record is given as input to the encoder (solid arrows). The decoder then predicts the optimal next values of the control parameters based on this data.

</div>

able solution that outperforms traditional methods used for quantum feedback control. Using transfer learning, we can demonstrate that the transformer also generalizes well to non-Markovian open quantum systems, which has yet to be demonstrated by existing methods.

Setup. We consider a quantum system undergoing a continuous weak measurement of the diffusive kind, e.g. homodyne readout in quantum optics [20, 21] or electronic charge detection by a quantum point contact [22, 23]. Let $ \hat{\rho}_{t} $ denote the state of the system at time t, conditioned on the measurement record $ \mathbf{r}_{t} $ . An experimenter uses their knowledge of the measurement record to control the system by manipulating some control parameter $ \lambda_{t} $ entering its Hamiltonian $ \hat{H}(\lambda_{t}) $ . The conditional dynamics is then described by a stochastic master equation of the form [4]

$$
d \hat {\rho} _ {t} = \frac {1}{i \hbar} [ \hat {H} \left(\lambda_ {t}\right), \hat {\rho} _ {t} ] d t + \mathcal {D} [ \hat {c} ] \hat {\rho} _ {t} d t + \sqrt {\eta} \mathcal {H} [ \hat {c} ] \hat {\rho} _ {t} d W _ {t},
$$

where the jump operator c describes the effect of coupling to the measuring device, the dissipation superoperator is $ \mathcal{D}[\hat{c}]\hat{\rho}=\hat{c}\hat{\rho}\hat{c}^{\dagger}-\frac{1}{2}\left(\hat{c}^{\dagger}\hat{c}\hat{\rho}+\hat{\rho}\hat{c}^{\dagger}\hat{c}\right) $ , the innovation superoperator is $ \mathcal{H}[\hat{c}]\hat{\rho}=\hat{c}\hat{\rho}+\hat{\rho}\hat{c}^{\dagger}-\operatorname{Tr}\left[\left(\hat{c}+\hat{c}^{\dagger}\right)\hat{\rho}\right]\hat{\rho} $ , the measurement efficiency is $ \eta $ , and the measurement noise in each small time step dt is described by independent Wiener increments $ dW_{t} $ with zero mean and variance $ dW_{t}^{2}=dt $ [24]. The measurement record $ r_{t} $ increments according to

$$
d r _ {t} = \operatorname {T r} \left[ \left(\hat {c} + \hat {c} ^ {\dagger}\right) \hat {\rho} _ {t} \right] d t + \frac {d W _ {t}}{\sqrt {\eta}}.
$$

Note that we use boldface notation to distinguish the history of the measurement record up to time $ t $ , $ \mathbf{r}_{t}= $ $ (\cdots,r_{t-2dt},r_{t-dt},r_{t}) $ , from its instantaneous value, $ r_{t}. $

For simplicity, we consider a single jump operator c and control parameter $ \lambda_{t} $ , but our method can be generalized straightforwardly to the case of multiple jump operators and control parameters.

In a general feedback protocol, the control parameter for the next time step is determined by the entire past history of the measurement record, i.e., it is a functional $ \lambda_{t+dt}[\mathbf{r}_t] $ . In the simplest case of linear feedback [20], the control parameter is proportional to the measurement result, $ \lambda_{t+dt}\propto dr_{t} $ , but this approach permits a very limited class of protocols and also suffers badly from measurement inefficiencies [25, 26]. More general state-based methods [27] decide the optimal feedback using a (implicit or explicit) model of the conditional quantum state, e.g. by solving Eq. (1) using experimentally obtained values for the measurement noise $ dW_{t} $ . Alternatively, reinforcement learning creates an implicit model of the dynamics in terms of a probability distribution (policy function) $ \pi_{\theta}(\lambda_{t+dt}|\mathbf{r}_{t}) $ , which is represented as a neural network parametrized by some weights and biases $ \theta $ . While recent successes of this approach have been demonstrated using the RNN architecture [5, 10], here we take a different approach based on the transformer architecture [15].

Transformer model. Our model consists of a custom transformer encoder-decoder architecture (see Fig. 1), which we name QuantumEncoder and QuantumDecoder, to determine $ \lambda_{t} $ at each time step. At its core, a transformer is designed to process sequential data by capturing long-range dependencies and contextual information. Unlike traditional recurrent neural networks (RNNs) that process sequences step by step, transformers employ a mechanism called self-attention to attend to different parts of the input simultaneously.

The QuantumEncoder processes the initial quantum state and the measurement record, embedding it into a higher-dimensional space and capturing dependencies through self-attention mechanisms. The QuantumDecoder takes the measurement record, with positional embeddings, as its input since this detailed sequential or spatial information is not provided by the encoder's compressed representation of core features from its input (its latent space). By feeding the measurement record into the decoder, we allow the model to adaptively adjust the optimal control parameter at each time step based on the observed system dynamics. The decoder employs self-attention that is causally masked, preventing it from 'seeing' future measurements, to ensure that predictions for the optimal control parameters at each time step are based only on current and past measurements. During training, the QuantumDecoder module takes the optimal parameter values and measurement record as input and learns to predict the next value $ \lambda_{t} $ in the sequence. We use the fidelity between the evolved state due to the $ \lambda_{t} $ and target state as the loss function used to train our model. The output of the decoder's last layer is passed through a linear transformation followed by a softmax function to obtain a probability distribution over the $ \lambda_{t} $ values. We also set up a sweep to optimize for optimal number of layers, the learning rate, optimizer and the number of epochs (see Supplemental Material [28]).

State stabilization in the two-level system. To demonstrate the effectiveness of the transformer-based approach, we showcase a numerical example of quantum state stabilization. The loss function in this case is the infidelity between the conditional state and some pure target state $ |\psi_{\mathrm{targ}} \rangle $

$$
L = 1 - \left\langle \psi_ {\mathrm {t a r g}} \right| \hat {\rho} _ {t} | \psi_ {\mathrm {t a r g}} \rangle .
$$

Our model system comprises a two-level system (TLS) with Hamiltonian

$$
\hat {H} \left(\lambda_ {t}\right) = \frac {\hbar \varepsilon}{2} \hat {\sigma} _ {z} + \frac {\hbar \lambda_ {t}}{2} \hat {\sigma} _ {x},
$$

undergoing a continuous measurement described by the jump operator $ \hat{c}=\sqrt{\kappa}\hat{\sigma}_{-} $ . Here, $ \varepsilon $ denotes a fixed energy bias and $ \kappa $ denotes the measurement rate.

To train the neural network, we prepare a dataset consisting of several initial states of the two-level system that are then evolved using the smesolve method from the QuTIP python package used for simulating open quantum systems[29]. We then train our transformer using the data set consisting of a range of initial states, their associated measurement records, and a locally optimal control protocol, $ \lambda_{t} $ , that drives the system to the target state for each noise realization. This locally optimal control is found using the PaQS algorithm [27]. During the training phase, we always set $ \varepsilon=0. $

As seen in Fig. 2, the transformer generates feedback strategies that can stabilize the TLS in a coherent superposition state $ |\psi_{\mathrm{targ}}\rangle = (|0\rangle + i|1\rangle) / \sqrt{2} $ even with inefficient measurements (we take $ \eta=0. 7 $ ). The transformer approach is also robust against perturbations of the dynamics, as we show by introducing a significant bias $ \varepsilon\neq0 $ , which was absent during training. Further examples can be found in Supplemental Material [28].

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F2026080810235104a236c681074741%2Fcrop_1_1786155859478.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=sZ5IjlsKRoc%2BZhKMIsiTVp7P5k4%3D&Expires=1786760659' alt='OCR图片'/></div>

<div align="center">

FIG. 2. Fidelity $ \mathcal{F} $ with a target state as a function of time under feedback control. The initial state is $ \hat{\rho}_{0}=|\psi_{0}\rangle\langle\psi_{0}| $ where $ |\psi_{0}\rangle=\alpha|0\rangle+\beta|1\rangle $ with $ \alpha=\sqrt{\frac{7}{12}} $ and $ \beta=\sqrt{\frac{5}{12}} $ The target state is $ |\psi_{\mathrm{targ}}\rangle=\frac{1}{\sqrt{2}}(|0\rangle+i|1\rangle). $ The performance of the transformer under imperfect measurement efficiency $ (\eta=0.7) $ (blue, circles), and an increase in bias $ (\epsilon=0.5) $ (green, squares) is benchmarked against the fidelity improvement when randomly selecting $ \lambda_{t} $ values (orange, crosses).

</div>

Another beneficial feature of the transformer is the speed with which it outputs the optimal control parameter for the next time step. Table I compares the time taken to infer the entire trajectory by the transformer algorithm and the modified proportional and quantum state (PaQS) algorithm, where the latter requires solving the stochastic master equation (1) at each time step. We observe a speed-up of approximately two orders of magnitude in our numerical tests, which were performed using a standard laptop. This is because PaQS, and other iterative numerical solvers require multiple evaluations of the right-hand side of Eq. (1) per gradient evaluation and per line-search, whereas our deployed transformer avoids online integration of the stochastic master equation altogether. However, it should be noted that this speed advantage comes at the cost of a large memory required to store the neural network representation. This memory requirement is likely to prove the most significant bottleneck for integrating the transformer into optimized hardware such as GPUs or FPGAs [8].

In order to provide another example to demonstrate the flexibility and generalizability of our transformerbased approach, we apply it to the challenging problem of controlling non-Markovian quantum dynamics [30]. Specifically, we now consider our TLS to be coupled to a harmonic oscillator mode with angular frequency $ \Omega $ and

coupling strength g, leading to the Hamiltonian

$$
\hat {H} \left(\lambda_ {t}\right) = \frac {\hbar \varepsilon}{2} \hat {\sigma} _ {z} + \frac {\hbar \lambda_ {t}}{2} \hat {\sigma} _ {x} + \hbar \Omega \hat {a} ^ {\dagger} \hat {a} + \hbar g \hat {\sigma} _ {z} \left(\hat {a} + \hat {a} ^ {\dagger}\right).
$$

We assume that the oscillator mode is coupled to a broadband environment that is continuously monitored via homodyne detection, leading to a stochastic master equation of the form (1) with $ \hat{c}=\sqrt{\kappa}\hat{a} $ . This situation can be realized, for example, by a superconducting qubit interfaced with a cavity resonator that is itself coupled to a waveguide [31]. This situation is well known to lead to non-Markovian dynamics for the qubit when the cavity linewidth $ \kappa $ is not too large, i.e. if $ \kappa\lesssim g $ [32].

Alternatively, one can interpret the cavity mode as a "reaction coordinate" (RC), which represents a collective mode of a structured reservoir whose spectral density is peaked at frequency $ \Omega $ [33, 34]. The extended open quantum system comprising the TLS and RC can thus be understood as a Markovian embedding of the original non-Markovian dynamics induced by the structured reservoir [35]. Meanwhile, the residual (broadband) environment represents far-field degrees of freedom that can be monitored without disrupting the non-Markovian character of the TLS evolution.

Since non-Markovian effects can affect the system dynamics in a much longer time horizon, the attentionbased transformer model seems ideal to control nonMarkovian systems due to the self and multihead attention. We use transfer learning to fine-tune the transformer to predict optimal $ \lambda_{t} $ values for the reaction-coordinate setting with a smaller dataset. As seen in Fig. 3, the transformer learns to predict optimal $ \lambda_{t} $ even in this non-Markovian setting. To benchmark the transformer, we train a vanilla recurrent neural network (RNN) and a gated recurrent unit recurrent neural network (GRU-RNN) with up to 60 time steps of the measurement record [36]. We train the RNNs on the given number of time steps to avoid the vanishing gradient problem as explored in the literature of deep learning [37]. We can observe that even though the vanilla RNN and GRU-RNN perform slightly better than the transformer for shorter time periods, where the measurement record provided is much smaller, the transformer outper-

<table border="1"><tr><td>Method</td><td>Inference Time(in sec)</td></tr><tr><td>Hamiltonian Modified PaQS</td><td>19.05</td></tr><tr><td>Quantum Transformer Inference</td><td>0.23</td></tr></table>

TABLE I. The inference time in seconds of predicting optimal $ \lambda_{t} $ for a single trajectory with 100 discretized time steps during the evolution of the state governed by the stochastic master equation(1). We benchmark the inference time of the transformer against the time taken to calculate the optimal feedback operation using a gradient based solver for the PaQS approach. The inference is run on a 2021 Macbook Pro with 16GB of RAM and a 8-core CPU.

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F2026080810235104a236c681074741%2Fcrop_1_1786155859486.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=bxsZmJDb0e7ZjIJDAmLYLzWmkk0%3D&Expires=1786760659' alt='OCR图片'/></div>

<div align="center">

FIG. 3. The fidelity $ \mathcal{F} $ as a function of time while benchmarking the performance of the transformer (red, crosses) as compared to a vanilla recurrent neural network (green, circles) and a gated-recurrent unit recurrent neural network GRU-RNN (yellow, stars). The context of 2000 measurement record samples is provided in the case of the non-Markovian setting accounted for by the reaction coordinate embedding. The coupling with the bath provided by $ g=0.5 $ . The dimension of the reaction coordinate is truncated to $ d=6 $

</div>

forms the vanilla RNN and the GRU-RNN in the case of long context windows and non-Markovian closed-loop feedback control. This is likely due to the fact that the transformer can process and attend to arbitrary measurement record lengths and does not suffer from the requirement of a sequential approach such as from RNNs.

Many-body state preparation. In the preceding examples, we demonstrated that our transformer can be trained via supervised learning, using known locally optimal protocols such as PaQS. We now turn to the example of approximate ground-state preparation of a many-body system via feedback control [38], where no such baseline is available and our transformer can no longer be trained in a supervised manner [39]. Consider an open N-qubit mixed-field Ising chain governed by the Hamiltonian

$$
\hat {H} \left(\lambda_ {t}\right) = \lambda_ {t} \sum_ {i = 1} ^ {N} \hat {\sigma} _ {i} ^ {z} + g \sum_ {i = 1} ^ {N - 1} \hat {\sigma} _ {i} ^ {x} \hat {\sigma} _ {i + 1} ^ {x} + h \sum_ {i = 1} ^ {N} \hat {\sigma} _ {i} ^ {x},
$$

where g is the nearest-neighbor coupling strength, h is a static longitudinal field, and the transverse field $ \lambda_{t} $ is the control parameter. Our goal is to sweep this parameter from an initial value $ \lambda_{0}\approx 0 $ for which the tensorproduct ground state is generally easy to prepare—to a final value $ \lambda_{T}\gtrsim g $ , within a fixed time T, while ideally leaving the system close to the ground state of the final Hamiltonian. This is generally challenging in a finite-time protocol due to the creation of non-adiabatic excitations [40, 41]. Moreover, the Hamiltonian in Eq. (6) is non-integrable [42, 43] and thus finding locally optimal control solutions becomes unfeasible as N grows larger.

The feedback is now conditioned on a homodyne measurement of the collective spin operator $ \hat{c}=\sqrt{\kappa}\sum_{i}\hat{\sigma}_{i}^{z}, $

<table border="1"><tr><td>Spins</td><td>Ground State Energy</td><td>Mean Predicted Energy</td></tr><tr><td>4</td><td>-6.518g</td><td>-6.209g</td></tr><tr><td>6</td><td>-9.723g</td><td>-9.666g</td></tr><tr><td>8</td><td>-13.155g</td><td>-12.845g</td></tr></table>

<div align="center">

TABLE II. Comparison of ground-state and predicted energies for different number of spins. We set g = 6.07 $ \lambda_{0}=-0.097g $ $ \lambda_{T}=1.27g $ $ h=1.19g $ $ \kappa=0.148g $ and $ gT=5.56. $

</div>

which produces diffusive dynamics as given by Eq. (1), and an additional measurement of the mean energy $ \mathrm{tr}[\hat{H}(\lambda_{T})\hat{\rho}_{T}] $ at the final time. The latter provides the loss function to be minimised by the control protocol for a fixed time T and endpoints $ \{\lambda_{0},\lambda_{T}\} $ . Crucially, since labeled optimal trajectories are now absent, we utilize a model-free reinforcement learning type approach [17], that allows our model to learn optimal control pulses through trial and error by direct interaction and observation of rewards. We use the strategy of an online iteratively-refined decision transformer (IR-DT) training mechanism [44]. Our inputs remain consistent with the previous examples; however, we also include the return-to-go award based on the final energy. Table II shows the mean final energies reached by our trained transformer, demonstrating good performance even as the system size grows, e.g. we observe a maximum discrepancy with the true ground-state energy of 2% for 8 spins. These results demonstrate the flexibility of our transformer architecture, as it can be adapted to situations where model-based supervised learning is challenging or impossible.

Conclusion- In this work, we have presented a novel approach for closed-loop adaptive feedback control of open quantum systems using attention-based transformer neural networks. We have demonstrated that our quantum transformer model can effectively learn to predict optimal control parameters based on the initial state and measurement record of a two-level quantum system. The transformer architecture, with its self-attention mechanism, enables capturing long-range dependencies in the measurement record, outperforming traditional methods like recurrent neural networks for feedback control in non-Markovian systems where different temporal aspects of the measurement record may affect the state evolution. Due to these reasons, transformer models are an ideal candidate for applications in quantum prediction [45] as well as control which we have demonstrated in this work.

We have shown the robustness and scalability of our approach under various conditions, such as imperfect measurement efficiency and perturbations in the Hamiltonian. Furthermore, using transfer learning, we have successfully applied our transformer model to the challenging task of controlling non-Markovian open quantum systems, which is a significant advancement in the field. The attention-based approach to quantum feedback control offers several advantages, including faster inference times compared to state-of-the-art methods like the Hamiltonian Modified PaQS. The transformer's ability to handle long context windows and its scalability make it a promising tool for controlling complex quantum systems. Furthermore, we demonstrated the transformer's ability to perform reinforcement learning when optimal parameters for supervised training are inaccessible, highlighting the promise of our approach for tackling challenging many-body control problems. In conclusion, our work demonstrates the effectiveness of attention-based transformer models for closed-loop feedback control of Markovian, non-Markovian, and many-body open quantum systems. This approach opens up new possibilities for the development of robust and efficient quantum control techniques, which are crucial for the advancement of quantum technologies. Future work could explore the application of this method to even more complex quantum systems and investigate its performance in experimental settings.

Acknowledgements. The authors acknowledge useful discussions with Prof. Gerard Milburn. N.A. acknowledges support from the European Research Council (grant agreement 948932) and the Royal Society (URFR1-191150). M.T.M. is supported by a Royal Society University Research Fellowship. The research of F.M. is partially supported by the Munich Quantum Valley, which is supported by the Bavarian state government with funds from the Hightech Agenda Bayern Plus. This project is co-funded by the European Union and UK Research & Innovation (Quantum Flagship project ASPECTS, Grant Agreement No. 101080167). Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union, Research Executive Agency or UK Research & Innovation. Neither the European Union nor UK Research & Innovation can be held responsible for them. P.V. is supported by the United States Army Research Office under Award No. W911NF-21S-0009-2. The authors would like to acknowledge the use of the University of Oxford Advanced Research Computing (ARC) facility in carrying out this work. http://dx.doi.org/10.5281/zenodo.22558.

Data Availability. The data associated with model architecture, training and data generation are publicly available [46]. Due to the size of the dataset generated for training and the hosting constraints, the generated data is available upon request.

quardt, Boosting the gottesman-kitaev-preskill quantum error correction with non-markovian feedback (2023), arXiv:2312.07391 [quant-ph].

[3] H. M. Wiseman, S. Mancini, and J. Wang, Physical Review A 66, 10.1103/physreva.66.013807 (2002).

[4] H. M. Wiseman and G. J. Milburn, Quantum Measurement and Control, 1st ed. (Cambridge University Press, 2009).

[5] T. Fösel, P. Tighineanu, T. Weiss, and F. Marquardt, Phys. Rev. X 8, 031084 (2018).

[6] Z. T. Wang, Y. Ashida, and M. Ueda, Phys. Rev. Lett. 125, 100401 (2020).

[7] V. V. Sivak, A. Eickbusch, H. Liu, B. Royer, I. Tsioutsios, and M. H. Devoret, Phys. Rev. X 12, 011059 (2022).

[8] K. Reuer, J. Landgraf, T. Fösel, J. O'Sullivan, L. Beltran, A. Akin, G. J. Norris, A. Remm, M. Kerschbaum, J.-C. Besse, F. Marquardt, A. Wallraff, and C. Eichler, Nature Communications 14, 7138 (2023).

[9] S. Borah, B. Sarma, M. Kewming, G. J. Milburn, and J. Twamley, Phys. Rev. Lett. 127, 190403 (2021).

[10] R. Porotti, V. Peano, and F. Marquardt, PRX Quantum 4, 030305 (2023).

[11] Y. Bengio, P. Simard, and P. Frasconi, IEEE transactions on neural networks 5, 157 (1994).

[12] E. Genois, J. A. Gross, A. Di Paolo, N. J. Stevenson, G. Koolstra, A. Hashim, I. Siddiqi, and A. Blais, PRX Quantum 2, 040355 (2021).

[13] H.-P. Breuer, E.-M. Laine, J. Piilo, and B. Vacchini, Rev. Mod. Phys. 88, 021002 (2016).

[14] G. T. Landi, M. J. Kewming, M. T. Mitchison, and P. P. Potts, PRX Quantum 5, 020201 (2024).

[15] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, Advances in neural information processing systems 30 (2017).

[16] U. Kamath, K. Graham, and W. Emara, Transformers for machine learning: a deep dive (Chapman and Hall/CRC, 2022).

[17] M. Janner, Q. Li, and S. Levine, Advances in neural information processing systems 34, 1273 (2021).

[18] L. Chen, K. Lu, A. Rajeswaran, K. Lee, A. Grover, M. Laskin, P. Abbeel, A. Srinivas, and I. Mordatch, Advances in neural information processing systems 34, 15084 (2021).

[19] K. Jacobs and D. A. Steck, Contemporary Physics 47, 279 (2006).

[20] H. M. Wiseman and G. J. Milburn, Phys. Rev. Lett. 70, 548 (1993).

[21] H. M. Wiseman, Phys. Rev. A 49, 2133 (1994).

[22] A. N. Korotkov, Phys. Rev. B 60, 5737 (1999).

[23] H.-S. Goan, G. J. Milburn, H. M. Wiseman, and H. Bi Sun, Phys. Rev. B 63, 125326 (2001).

[24] Y. Jiang, X. Wang, L. Martin, and K. B. Whaley, Phys. Rev. A 102, 022612 (2020).

[25] J. Wang and H. M. Wiseman, Phys. Rev. A 64, 063810 (2001).

[26] M. T. Mitchison, J. Goold, and J. Prior, Quantum 5, 500 (2021).

[27] S. Zhang, L. S. Martin, and K. B. Whaley, Phys. Rev. A 102, 062418 (2020).

[28] Supplemental material (2025), see Supplemental Material at placeholder for more examples and details on the model training, architecture and performance.

[29] J. Johansson, P. Nation, and F. Nori, Computer Physics

Communications 184, 1234 (2013).

[30] C.-F. Li, G.-C. Guo, and J. Piilo, Europhysics Letters 128, 30001 (2020).

[31] A. Blais, R.-S. Huang, A. Wallraff, S. M. Girvin, and R. J. Schoelkopf, Phys. Rev. A 69, 062320 (2004).

[32] H.-P. Breuer and F. Petruccione, The theory of open quantum systems (OUP Oxford, 2002).

[33] J. Iles-Smith, N. Lambert, and A. Nazir, Phys. Rev. A 90, 032114 (2014).

[34] D. Tamascelli, A. Smirne, S. F. Huelga, and M. B. Plenio Phys. Rev. Lett. 120, 030402 (2018).

[35] M. P. Woods, R. Groux, A. W. Chin, S. F. Huelga, and M. B. Plenio, Journal of Mathematical Physics 55, 032101 (2014), https://pubs.aip.org/aip/jmp/article-pdf/doi/10.1063/1.4866769/14759880/032101_1_online.pdf

[36] K. Cho, B. van Merrienboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, Learning phrase representations using rnn encoder-decoder for statistical machine translation (2014), arXiv:1406.1078 [cs.CL].

[37] R. Pascanu, T. Mikolov, and Y. Bengio, in Proceedings of the 30th International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 28, edited by S. Dasgupta and D. McAllester (PMLR, Atlanta, Georgia, USA, 2013) pp. 1310-1318.

[38] F. Metz and M. Bukov, Nature Machine Intelligence 5, 780 (2023).

[39] Y.-H. Zhang and M. Di Ventra, Physical Review B 107, 10.1103/physrevb.107.075147 (2023).

[40] M. Kolodrubetz, D. Sels, P. Mehta, and A. Polkovnikov, Physics Reports Geometry and Non-Adiabatic Response in Quantum and Classical Systems, 697, 1 (2017).

[41] D. Guery-Odelin, A. Ruschhaupt, A. Kiely, E. Torrontegui, S. Martínez-Garaot, and J. G. Muga, Rev. Mod. Phys. 91, 045001 (2019).

[42] H. Kim and D. A. Huse, Phys. Rev. Lett. 111, 127205 (2013).

[43] J. F. Rodriguez-Nieva, C. Jonay, and V. Khemani, Phys. Rev. X 14, 031014 (2024).

[44] Q. Zheng, A. Zhang, and A. Grover, in international conference on machine learning (PMLR, 2022) pp. 27042- 27059.

[45] L. E. H. Rodriguez and A. A. Kananenka, A short trajectory is all you need: A transformer-based model for long-time dissipative quantum dynamics (2024), arXiv:2409.11320 [quant-ph].

[46] Code for the network architecture and data generation (2025), access to github link: https://github.com/ pranavjv/transformerfeedbackcontrol.

[47] S. J. Wright, Numerical optimization (2006).

[48] A. Radford, K. Narasimhan, T. Salimans, I. Sutskever, et al., (2018).

[49] W. Li, H. Luo, Z. Lin, C. Zhang, Z. Lu, and D. Ye, A survey on transformers in reinforcement learning (2023), arXiv:2301.03044 [cs.LG].

[50] L. Liu, H. Jiang, P. He, W. Chen, X. Liu, J. Gao, and J. Han, On the variance of the adaptive learning rate and beyond (2021), arXiv:1908.03265 [cs.LG].

[51] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, Proximal policy optimization algorithms (2017), arXiv:1707.06347 [cs.LG].

## Appendix A: Local Optimality Using the PaQS Approach

In this appendix, we summarize the PaQS approach of Zhang et al. [27] as applied to our problem of interest. For simplicity, we set the bias $ \varepsilon=0 $ and focus on the effect of varying $ \lambda_{t} $ to control the system. We consider an arbitrary target state:

$$
\left| \psi_ {T} \right\rangle = \left| \psi_ {\mathrm {s y s t e m}} \right\rangle \otimes \left| \psi_ {\mathrm {R C}} \right\rangle
$$

for some generic state $ |\psi_{\mathrm{system}} \rangle $ and $ |\psi_{\mathrm{RC}} \rangle $ of the reaction coordinate. As explained in the main text, the role of the reaction coordinate is essential for performing the Markovian embedding of a non-Markovian system. When solving the optimal control of the system, the combined state of the system and reaction coordinate is treated collectively in the density matrix, allowing standard Markovian master equation techniques to be applied to this extended system. However, it makes the analysis somewhat more complicated than Ref. [27] because of the presence of the Hamiltonian of the reaction coordinate, in addition to the control Hamiltonian proportional to $ \lambda_{t} $

To proceed, we separate the evolution with and without feedback. In the absence of feedback, the state evolves over a timestep dt according to the stochastic master equation $ \left( \hbar=1\right) $

$$
d \hat {\rho} _ {t} = - i [ \hat {H} _ {0}, \hat {\rho} _ {t} ] d t + \mathcal {D} [ \hat {c} ] \hat {\rho} _ {t} d t + \sqrt {\eta} \mathcal {H} [ \hat {c} ] \hat {\rho} _ {t} d W _ {t},
$$

with the Hamiltonian $ \hat{H}_{0}\equiv\hat{H}(\lambda_{t}=0). $ Following Zhang et al., we describe the feedback control by the unitary operator

$$
\hat {U} \left(\theta_ {t}\right) \equiv e ^ {- i \theta_ {t} \hat {H} _ {F}},
$$

where $ \hat{H}_{F}=\sigma_{x}/2 $ is the feedback Hamiltonian and the infinitesimal rotation angle $ \theta_{t}=\lambda_{t}dt $ encapsulates the effect of the control parameter. In the following analysis, for the sake of simplicity, we consider the measurement efficiency $ \eta=1 $ . The fidelity with respect to a target state $ |\psi_{T}\rangle $ is $ \mathcal{F}_{t}=\langle\psi_{T}|\hat{\rho}_{t}|\psi_{T}\rangle $ , which thus updates according to

$$
\mathcal {F} _ {t + d t} = \langle \psi_ {T} | \hat {U} \left(\theta_ {t}\right) \left[ \hat {\rho} _ {t} + d \hat {\rho} _ {t} \right] \hat {U} ^ {\dagger} \left(\theta_ {t}\right) | \psi_ {T} \rangle .
$$

The goal of locally optimal control is to choose the rotation angle $ \theta_{t} $ to maximize the fidelity with the target state at each step. We therefore demand $ \mathcal{G}=0 $ , where the cost function is

$$
\mathcal {G} \equiv \frac {\partial \mathcal {F} _ {t + d t}}{\partial \theta_ {t}} = - i \left\langle \psi_ {T} \left| \left[ \hat {H} _ {F}, \hat {\rho} _ {t} \right] \right| \psi_ {T} \right\rangle + \mathcal {O} (d t),
$$

and we keep only the leading-order term, neglecting infinitesimal corrections. We denote by $ \theta_{t}^{*} $ the optimal value of $ \theta_{t} $ that solves $ \mathcal{G}=0 $ . Since $ \theta_{t}^{*} $ is infinitesimal, it can be parameterized without loss of generality as [27]

$$
\theta_ {t} ^ {*} = A _ {1} (t) d W _ {t} + A _ {2} (t) d t,
$$

where $ A_{1} $ and $ A_{2} $ are to be solved for.

To get explicit expressions for the functions $ A_{1} $ and $ A_{2} $ , we expand the unitary operator $ U_{F} $ up to second order in $ dW_{t} $ and make use of the rules of Ito calculus, i.e., $ dW^{2}=dt $ and $ dWdt=0=dt^{2} $ . The second-order expansion of $ U(\theta_{t}^{*}) $ is given by

$$
U = I - i A _ {1} \hat {H} _ {F} d W - \left(i A _ {2} \hat {H} _ {F} + \frac {1}{2} A _ {1} ^ {2} \hat {H} _ {F} ^ {2}\right) d t.
$$

Substituting this into the state update rule $ \hat{\rho}_{t+dt}=\hat{U}(\theta_{t}^{*})[\hat{\rho}_{t}+d\hat{\rho}_{t}]\hat{U}^{\dagger}(\theta_{t}^{*}) $ and simplifying yields

$$
\begin{array}{l} \hat {\rho} _ {t + d t} = \hat {\rho} _ {t} + d t \left(- i \left[ \hat {H} _ {0}, \hat {\rho} _ {t} \right] + \kappa \mathcal {D} [ \hat {a} ] \rho - i A _ {2} \left[ \hat {H} _ {F}, \hat {\rho} _ {t} \right] - \frac {1}{2} A _ {1} ^ {2} \left\{\hat {H} _ {F} ^ {2}, \hat {\rho} _ {t} \right\} \\ + A _ {1} ^ {2} \hat {H} _ {F} \hat {\rho} _ {t} \hat {H} _ {F} - i A _ {1} \sqrt {\kappa} \left[ \hat {H} _ {F}, \mathcal {H} [ \hat {a} ] \hat {\rho} _ {t} \right]) \\ d W _ {t} \left(- i A _ {1} \left[ \hat {H} _ {F}, \hat {\rho} _ {t} \right] + \sqrt {\kappa} \mathcal {H} [ \hat {a} ] \hat {\rho} _ {t}\right). \\ \end{array}
$$

We can now substitute this expression into the cost function in Eq. (11) to find

$$
\mathcal {G} = - i \left\langle \psi_ {T} \left| \begin{array}{c} \hat {\rho} _ {t} - i A _ {2} d t \left[ \hat {H} _ {F}, \hat {\rho} _ {t} \right] - i A _ {1} d W \left[ \hat {H} _ {F}, \hat {\rho} _ {t} \right] + d t \left(- i \left[ \hat {H} _ {0}, \hat {\rho} _ {t} \right] + \kappa \mathcal {D} [ a ] \hat {\rho} _ {t} + A _ {1} ^ {2} \mathcal {D} \left[ \hat {H} _ {F} \right] \hat {\rho} _ {t} \right. \\ - i A _ {1} \sqrt {\kappa} \left[ \hat {H} _ {F}, \mathcal {H} [ a ] \hat {\rho} _ {t} \right]) + \sqrt {\kappa} \mathcal {H} [ a ] \hat {\rho} _ {t} d W \end{array} \right| \psi_ {T} \rangle = 0.
$$

Since this cost is defined for the target state as mentioned in Eq. (7) which consists of $ |\psi_{\mathrm{RC}}\rangle $ . We can then solve Eq. (15) in a truncated Hilbert space numerically using algorithms such as the Broyden-Fletcher-Goldfarb-Shanno (BFGS) or the modified Newton-Raphson algorithm to find the optimal $ A_{1} $ and $ A_{2} $ [47]. Following a similar analysis to solving $ \mathcal{G} $ analytically, we get:

$$
A _ {1} = \frac {\sqrt {\kappa} \left\langle \psi_ {T} \mid \mathcal {H} [ \hat {a} ] \hat {\rho} _ {t} \mid \psi_ {T} \right\rangle}{i \left\langle \psi_ {T} \left| \left[ \hat {H} _ {F}, \hat {\rho} _ {t} \right] \right| \psi_ {T} \right\rangle}
$$

and

$$
A _ {2} = \frac {\left\langle \psi_ {T} \right| \left(- i \left[ \hat {H} _ {0} , \hat {\rho} _ {t} \right] + \kappa \mathcal {D} [ \hat {a} ] \hat {\rho} _ {t} + A _ {1} ^ {2} \mathcal {D} \left[ \hat {H} _ {F} \right] \hat {\rho} _ {t} - i A _ {1} \sqrt {\kappa} \left[ \hat {H} _ {F} , \mathcal {H} [ \hat {a} ] \hat {\rho} _ {t} \right]\right) \left| \psi_ {T} \right\rangle}{i \left\langle \psi_ {T} \right| \left[ \hat {H} _ {F}, \hat {\rho} _ {t} \right] \left| \psi_ {T} \right\rangle}
$$

## Appendix B: Attention Formalism

Transformers are inherently permutation-invariant due to their parallel processing of input sequences, which poses a challenge when dealing with sequential data where order is crucial. To address this, we incorporate positional embeddings into the measurement record input. Positional embeddings encode temporal relationships between different parts of the measurement record, allowing the model to capture the sequence's temporal dynamics. This encoding ensures that the model can distinguish between measurements taken at different time steps, which is essential for accurately predicting the optimal control parameters over time [15].

The encoder's output serves as a context vector summarizing the initial state and measurement record information. The self-attention mechanism can be conceptualized as a graph-like structure, where each element in the input sequence is connected to every other element. The strength of these connections, or attention weights, is learned through training. This allows the model to weigh the importance of different parts of the input when generating an output. Mathematically, the self-attention mechanism can be described as a weighted sum of value vectors, where the weights are determined by the compatibility between query and key vectors. These vectors are obtained by applying learned linear transformations to the input embeddings. The QuantumDecoder module is structured similarly to a generative pretrained transformer (GPT) module along with positional embeddings to perform an autoregressive task [48]. During training, the QuantumDecoder module which consists of embedding layers for the optimal parameter values and measurement record, followed by a Transformer decoder takes the optimal parameter values, context (encoded representation from the encoder), and measurement record as input and predicts the next value $ \lambda_{t} $ in the sequence.

In our transformer-based model for quantum control, the attention mechanism is important to capture the complex dependencies between the initial quantum state, the sequential measurement records, and the optimal control parameters that we aim to predict. Here, we focus on a detailed, yet concise explanation of how attention operates within our model, considering both selfattention and cross-attention. The cross-attention layers in the decoder enable it to integrate contextual information from the encoder.

Cross-attention bridges the encoder and decoder by computing attention scores between the decoder's queries and the encoder's keys and values. Several attention scores can be calculated by different linear transformation known as attention heads. Multiple attention heads operate in parallel, and their outputs are concatenated and transformed. Feed-forward networks introduce non-linearity and residual connections and layer normalization are applied for stable training. We use multiple encoder modules together to learn hierarchical representations as well. By stacking several identical Transformer-encoder blocks—each comprising multihead self-attention, a feed-forward network, and residual normalization—the model builds depth that progressively abstracts the input. Early layers focus on very local measurement correlations, mid-level layers capture medium-range dependencies, and the deepest layers integrate global structure needed for long-horizon planning. This leads to better out-of-distribution generalization as well.

While reinforcement learning methods can learn control strategies through interaction with the environment, they often require extensive exploration and can suffer from instability during training. Our transformer-based approach provides a stable and efficient alternative by learning from supervised data. It eliminates the need for exploration by utilizing known optimal control parameters during training. This approach leverages the strengths of sequence modeling inherent in transformers, capturing long-range dependencies without the high variance typically associated with RL methods [49].

Self-attention allows the model to weigh the relevance of different elements within a single input sequence by computing attention scores between all pairs of positions. For each element in the sequence, the model generates query vectors (Q), key vector (K), and value vector (V) using learned linear transformations.

$$
\begin{array}{l} Q = W _ {Q} \cdot X \\ K = W _ {K} \cdot X \\ V = W _ {V} \cdot X \\ \end{array}
$$

where X is the latent space embeddings from the input and $ W_{Q} $ , $ W_{K} $ and $ W_{V} $ are learned weight vectors. The self attention can be calculated as:

$$
\operatorname {A t t e n t i o n} (Q, K, V) = \operatorname {s o f t m a x} \left(\frac {Q K ^ {T}}{\sqrt {d _ {k}}}\right) V.
$$

When we include both the initial quantum state and

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F2026080810235104a236c681074741%2Fcrop_1_1786155859491.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=EwZlWIYMpoUmwAZdsi4QqmMWmZc%3D&Expires=1786760659' alt='OCR图片'/></div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F2026080810235104a236c681074741%2Fcrop_2_1786155859496.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=5G7zu8GGpvfE1pPcMgZ8zcVK0dQ%3D&Expires=1786760659' alt='OCR图片'/></div>

<div align="center">

(b)

</div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F2026080810235104a236c681074741%2Fcrop_3_1786155859500.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=9ssI5m5js6%2Bwjt6F8epwoL4Qjx8%3D&Expires=1786760659' alt='OCR图片'/></div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F2026080810235104a236c681074741%2Fcrop_4_1786155859505.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=IUTywkuq9VXk8AK02SjhPq88XFM%3D&Expires=1786760659' alt='OCR图片'/></div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F2026080810235104a236c681074741%2Fcrop_5_1786155859510.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=ESf8lec3wtffVsWhY%2BrN2Vlc%2BWk%3D&Expires=1786760659' alt='OCR图片'/></div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F2026080810235104a236c681074741%2Fcrop_6_1786155859515.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=5cfRY1yG%2BNJkqRoPbdO9aoVGsac%3D&Expires=1786760659' alt='OCR图片'/></div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F2026080810235104a236c681074741%2Fcrop_7_1786155859520.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=gXAlM1nlyvioOvPGzpXY1LLIHIA%3D&Expires=1786760659' alt='OCR图片'/></div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F2026080810235104a236c681074741%2Fcrop_8_1786155859538.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=hAQ%2FRYWAtsxeeRn%2FP%2Fi6NKYHu1A%3D&Expires=1786760659' alt='OCR图片'/></div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F2026080810235104a236c681074741%2Fcrop_9_1786155859543.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=FrHPVaEBjac40Rq8BZXu1xvQxfc%3D&Expires=1786760659' alt='OCR图片'/></div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F2026080810235104a236c681074741%2Fcrop_10_1786155859549.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=xaLGGyjupbJniTxFhFQau8JFz6U%3D&Expires=1786760659' alt='OCR图片'/></div>

<div align="center">

FIG. 4. a, The magnitude and phase map of the mixed initial state of $ \rho_{0}=0. 7 | 0 \rangle\langle 0 |+0. 3 | 1 \rangle\langle 1 | $ and target pure state of $ | \psi_{T} \rangle=\sqrt{0. 3}|0 \rangle+i \sqrt{0. 7}|1 \rangle $ . The third plot represents the increase in fidelity towards the target pure state based on control parameters produced by the transformer (blue, circles) when undergoing continuous measurement with a measurement efficiency of 0.8. b, The magnitude and phase map of the mixed initial state of $ \rho_{0}=0. 6 | 0 \rangle\langle 0 |+(0. 2+0. 1 i)|0 \rangle\langle 1 |+(0. 2-0. 1 i)|1 \rangle\langle 0 |+0. 4 | 1 \rangle\langle 1 | $ and target pure state of $ \psi_{T}=\frac{|0\rangle}{\sqrt{2}}+\frac{1+i}{2}|1\rangle $ . The third plot represents the increase in fidelity towards the target pure state based on control parameters produced by the transformer (blue, circles) when undergoing continuous measurement with a measurement efficiency of 0.65.

</div>

the measurement record in the encoder, self-attention computes how each element (state or measurement at a certain time) relates to every other element in the sequence. This enables the encoder to build a comprehensive contextual representation that captures relationships between the initial conditions and the observed dynamics.

The decoder processes the measurement record up to the current time step. Masked self-attention ensures that predictions for the control parameters at each time step only depend on current and past measurements, preserving causality. This mechanism allows the decoder to understand temporal dependencies within the measurement sequence.

The cross-attention occurs in the decoder and allows it to incorporate information from the encoder's output. The decoder's queries attend to the encoder's keys and

values, integrating contextual information from the encoder:

$$
\mathrm {A t t e n t i o n} = \mathrm {s o f t m a x} \left(\frac {Q ^ {\mathrm {d e c}} \left(K ^ {\mathrm {e n c}}\right) ^ {\top}}{\sqrt {d _ {k}}}\right) V ^ {\mathrm {e n c}}.
$$

## Appendix C: Model and Dataset Details

The model consists of 6 encoder and 6 decoder layers. Each layer comprises an embedding dimension of 512 and 8 attention heads. In order to maximize performance, a context window of 1024 tokens was chosen. Training was performed using the RAdam optimizer with an initial learning rate of 0.001 and was trained for 100 epochs with early stopping on validation loss to prevent overfitting [50]. Gradient clipping with a maximum norm of 1 was applied to stabilize training and prevent exploding

gradients. Model training was performed on a NVIDIA A100 GPU with 80GB of RAM.

The dataset was generated using the QuTiP package [29]. There were 200 unique initial quantum states generated. For each initial state, the smesolve method was used to simulate 1,000 stochastic trajectories using the stochastic master equation, with each trajectory consisting of 1,000 time steps.

## Appendix D: Examples

## Appendix E: Iteratively Refined Decision Transformer(IR-DT)

In this appendix, we present more examples to demonstrate the performance of the transformer further. We provide examples of state preparation and purification from a varied set of initial and target states as seen in Fig 4. We also demonstrate that the transformer predicts optimal control parameters for state purification and preparation even under measurement inefficiencies under continuous measurement.

The architecture we use for the reinforcement learning task of learning the ground state energy starts from a conventional Decision-Transformer. Because no expert demonstrations exist for the transverse-Ising task, we first pre-train the network on a modest offline buffer of random control strings together with the energies they

realize . Then an outer self-improvement loop is launched following these steps:

- Roll out the current transformer policy to generate M new trajectories.

- Re-label each trajectory with the actual energy return obtained at the end of the sweep.

- Aggregate these fresh state-action-return triples into the replay buffer.

- Fine-tune the same network on the enlarged dataset.

Because the Decision-Transformer conditions every decoder step on the desired return, newly added lowenergy examples automatically bias the next policy towards deeper minima while maintaining training stability. Causal masking enforces autoregressive conditioning, while return conditioning biases the output toward lower energies without recourse to a separate value network or policy gradient as seen in traditional reinforcement learning methods [51]. Repeating this refine-and-retrain cycle drives the trajectory distribution monotonically downward in energy without ever computing policy gradients or value functions, allowing the transformer to discover near-ground-state control schedules for chains up to N=8. Note, however, that this approach is extremely data intensive requiring more than >100k trajectories.