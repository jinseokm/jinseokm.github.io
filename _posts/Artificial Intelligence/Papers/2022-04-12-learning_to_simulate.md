---
title: "Learning to Simulate Complex Physics with Graph Networks"
categories:
  - Papers
tags: [GNN, Physics, ICML]
toc: true
toc_sticky: true
toc_label: "On this page"
published: true
use_math: true

date: 2022-04-12
last_modified_at: 2022-04-16
---

# [Learning to Simulate Complex Physics with Graph Networks (ICML 2020)](https://arxiv.org/abs/2002.09405)
DeepMind 에서는 Graph 와 Particle-based Simulation 을 결합한 `Graph Network-based Simulators (GNS)` 를 제안했습니다. 그들이 제안한 Learning to Simulate 모델을 이용해서 시뮬레이션을 모사하면 다음과 같이 유체의 흐름을 모사할 수 있습니다.

<center>
<figure style="width: 70%"> <img src="/Images/Study/learning_to_simulate/water_ramps_rollout.gif" alt="Learning to Simulate Example"/>
<figcaption>Learning to Simulate 모델 예시</figcaption>
</figure>
</center>

# Graph Neural Network (GNN)
먼저, 그래프가 무엇인지 알아보겠습니다. 그래프는 점들과 그 점들을 잇는 선으로 이루어진 데이터 구조이며, 관계나 상호작용을 나타내는 데이터를 분석할 때 주로 쓰입니다. 대표적인 예로는 페이스북 친구관계, 왓챠플레이(유튜브, 넷플릭스) 유저-영상 감상여부 등이 있습니다. 이외에도 Face recognition, pose estimation 등 다양한 방면으로 활용됩니다.

<center>
<figure style="width: 30%"> <img src="/Images/Study/learning_to_simulate/graph.png" alt="Exmaple of Graph"/>
<figcaption>점과 선으로 이루어진 그래프</figcaption>
</figure>

<figure style="width: 60%"> <img src="/Images/Study/learning_to_simulate/relations.jpg" alt="Relation Networks"/>
<figcaption>인간 관계도. Image by <a href="https://pixabay.com/users/gdj-1086657/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=3846597">Gordon Johnson</a> from <a href="https://pixabay.com/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=3846597">Pixabay</a></figcaption>
</figure>
</center>

GNN은 2009년에 제안된 기법(Scarselli et al., 2009)으로, 노드와 노드 사이의 관계, 엣지의 가중치 들을 고려하여 그래프로 네트워크를 구성하는 방법입니다. 

<center>
<figure style="width: 60%"> <img src="/Images/Study/learning_to_simulate/gnn.png" alt="GNN"/>
<figcaption>Graph Neural Network (Scarselli et al., 2009) </figcaption>
</figure>
</center>

# Graph Networks
Graph Networks 는 DeepMind 에서 제작한 python, tf 기반 GNN 라이브러리로, 그래프를 입력받고 마찬가지로 그래프를 출력하는 구성이며 입력 그래프는 엣지(E), 노드(V), 그리고 전역 파라미터(u)로 구성되어있고, 출력 그래프는 입력 그래프와 같은 형태이나 각 파라미터들을 업데이트한 상태가 됩니다. 또한, 노드, 엣지, 글로벌에 더해 Sender/Receiver 자료형을 가지고 있습니다.

<center>
<figure style="width: 80%"> <img src="/Images/Study/learning_to_simulate/graphnets.jpg" alt="GNN"/>
<figcaption>Graph Networks</figcaption>
</figure>
</center>

아래 그림과 같은 그래프가 있을때, `[Sender, Receiver]` 는 `[[0,0], [0,1], [1,0], [1,0]]` 와 같은 식으로 구성되어 있습니다. 각 원소의 인덱스는 엣지의 인덱스를 나타냅니다.

<center>
<figure style="width: 60%"> <img src="/Images/Study/learning_to_simulate/graph-example.jpg" alt="Graph Example"/>
<figcaption>Graph Example</figcaption>
</figure>
</center>

입출력 네트워크를 구성하는 여러가지 방법이 있습니다.

<center>
<figure style="width: 60%"> <img src="/Images/Study/learning_to_simulate/modules.jpg" alt="Graph Example"/>
<figcaption>Graph Networks Modules</figcaption>
</figure>
</center>

# Learning to Simulate 코드 리뷰
[Github](https://github.com/deepmind/deepmind-research/tree/master/learning_to_simulate) 에 소스코드를 오픈하여 공유하고 있고, 파일별 역할은 다음과 같습니다.

> - train.py: 전체적으로 학습/평가/시뮬레이션 모사를 담당
> - learned_simluator.py: 한스텝을 학습, 다음 포지션을 예측. 데이터 전처리, 정규화  기능 포함
> - graph_network.py: 코어 네트워크 모델
> - render_rollout.py: 결과 가시화
> - {noise/connectivity/reading}_utils.py: 노이즈 첨가, 그래프 연결정보 구성 (주변입자 탐색), 데이터 셋 읽기 등의 유틸 기능
 
초기 세팅은 flags 를 이용해서 다음과 같이 정의하여 argument 들을 사용합니다.

```python
# train.py

flags.DEFINE_enum(
    'mode', 'train', ['train', 'eval', 'eval_rollout'],
    help='Train model, one step evaluation or rollout evaluation.')
flags.DEFINE_enum('eval_split', 'test', ['train', 'valid', 'test'],
                  help='Split to use when running evaluation.')
flags.DEFINE_string('data_path', None, help='The dataset directory.')
flags.DEFINE_integer('batch_size', 2, help='The batch size.')
flags.DEFINE_integer('num_steps', int(2e7), help='Number of steps of training.')
flags.DEFINE_float('noise_std', 6.7e-4, help='The std deviation of the noise.')
flags.DEFINE_string('model_path', None,
                    help=('The path for saving checkpoints of the model. '
                          'Defaults to a temporary directory.'))
flags.DEFINE_string('output_path', None,
                    help='The path for saving outputs (e.g. rollouts).')


FLAGS = flags.FLAGS

Stats = collections.namedtuple('Stats', ['mean', 'std'])

INPUT_SEQUENCE_LENGTH = 6  # So we can calculate the last 5 velocities.
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3
```

`LearnedSimulator` 는 시뮬레이터와 GNN을 엮어주는 클래스로, 이 클래스를 통해서 전체적인 학습/평가/시뮬레이션을 수행하게 됩니다.

```python
# learned_simluator.py

class LearnedSimulator(snt.AbstractModule):
  """Learned simulator from https://arxiv.org/pdf/2002.09405.pdf."""

  def __init__(
      self,
      num_dimensions,
      connectivity_radius,
      graph_network_kwargs,
      boundaries,
      normalization_stats,
      num_particle_types,
      particle_type_embedding_size,
      name="LearnedSimulator"):
    super().__init__(name=name)

    self._connectivity_radius = connectivity_radius
    self._num_particle_types = num_particle_types
    self._boundaries = boundaries
    self._normalization_stats = normalization_stats
    with self._enter_variable_scope():
      self._graph_network = graph_network.EncodeProcessDecode(
          output_size=num_dimensions, **graph_network_kwargs)

      if self._num_particle_types > 1:
        self._particle_type_embedding = tf.get_variable(
            "particle_embedding",
            [self._num_particle_types, particle_type_embedding_size],
            trainable=True, use_resource=True)
```

위 코드의 LearnedSimulator 클래스는 멤버 변수로 `_graph_network` 를 갖는데, 이 변수는 Encoder - Processor - Decoder 로 구성되는 코어 네트워크를 정의한 `EncodeProcessDecode` 클래스입니다. 클래스의 세부 네트워크는 다음과 같습니다.

```python
# graph_network.py

class EncodeProcessDecode(snt.AbstractModule):
  """Encode-Process-Decode function approximator for learnable simulator."""

  def __init__(
      self,
      latent_size: int,
      mlp_hidden_size: int,
      mlp_num_hidden_layers: int,
      num_message_passing_steps: int,
      output_size: int,
      reducer: Reducer = tf.math.unsorted_segment_sum,
      name: str = "EncodeProcessDecode"):

    super().__init__(name=name)

    self._latent_size = latent_size
    self._mlp_hidden_size = mlp_hidden_size
    self._mlp_num_hidden_layers = mlp_num_hidden_layers
    self._num_message_passing_steps = num_message_passing_steps
    self._output_size = output_size
    self._reducer = reducer

    with self._enter_variable_scope():
      self._networks_builder()
```

`_networks_builder()` 함수는 다음과 같습니다. 노드와 엣지를 MLP 로 엮어서 독립적인 Encoder 네트워크를 각각 만들고, `_num_message_passing_steps` 만큼의 process 네트워크를 구축합니다. 이후 process 과정을 통해 얻어진 latent vector들을 decoder 과정을 통해 초기 노드와 동일한 차원으로 만들어줍니다.

```python
# graph_network.py
# class EncodeProcessDecode(snt.AbstractModule):

    def _networks_builder(self):
    """Builds the networks."""

        def build_mlp_with_layer_norm():
        mlp = build_mlp(
            hidden_size=self._mlp_hidden_size,
            num_hidden_layers=self._mlp_num_hidden_layers,
            output_size=self._latent_size)
        return snt.Sequential([mlp, snt.LayerNorm()])

        # The encoder graph network independently encodes edge and node features.
        encoder_kwargs = dict(
            edge_model_fn=build_mlp_with_layer_norm,
            node_model_fn=build_mlp_with_layer_norm)
        self._encoder_network = gn.modules.GraphIndependent(**encoder_kwargs)

        # Create `num_message_passing_steps` graph networks with unshared parameters
        # that update the node and edge latent features.
        # Note that we can use `modules.InteractionNetwork` because
        # it also outputs the messages as updated edge latent features.
        self._processor_networks = []
        for _ in range(self._num_message_passing_steps):
        self._processor_networks.append(
            gn.modules.InteractionNetwork(
                edge_model_fn=build_mlp_with_layer_norm,
                node_model_fn=build_mlp_with_layer_norm,
                reducer=self._reducer))

        # The decoder MLP decodes node latent features into the output size.
        self._decoder_network = build_mlp(
            hidden_size=self._mlp_hidden_size,
            num_hidden_layers=self._mlp_num_hidden_layers,
            output_size=self._output_size)
```

# Encoder
각 입자들에 대한 Input $\mathbf{x}_i$ 는 다음과 같이 정의됩니다. $i$ 입자의 위치, 그리고 유한차분법으로 얻어지는 이전 5스텝의 속도 $\dot{\mathbf{p}}_i$ , 그리고 입자가 boundary 인지, 유체인지, 강체인지 등을 나타내는 입자의 속성 $\mathbf{f}_i$ 로 이루어져있습니다.

$$
\begin{align}
\mathbf{x}_i^{t_k} =& \left[ \mathbf{p}_i^{t_k}, \dot{\mathbf{p}}_i^{t_{k-C+1}}, ..., \dot{\mathbf{p}}_i^{t_{k}}, \mathbf{f}_i \right] \\
\mathbf{r}_{i,j} =& \left[ \left( \mathbf{p}_i-\mathbf{p}_j \right), \lVert \mathbf{p}_i - \mathbf{p}_j \rVert \right]
\end{align}
$$

점군 데이터에 대해서는 기본적으로 주변 데이터 탐색이 필요합니다. 이 논문에서는 encoder 에 입력할 그래프로 만들어주기 위해서 [scikit-learn의 K-D Tree 알고리즘](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html) 을 사용했습니다.

```python
# connectivity_utils.py

def _compute_connectivity(positions, radius, add_self_edges):
  """Get the indices of connected edges with radius connectivity.

  Args:
    positions: Positions of nodes in the graph. Shape:
      [num_nodes_in_graph, num_dims].
    radius: Radius of connectivity.
    add_self_edges: Whether to include self edges or not.

  Returns:
    senders indices [num_edges_in_graph]
    receiver indices [num_edges_in_graph]

  """
  tree = neighbors.KDTree(positions)
  receivers_list = tree.query_radius(positions, r=radius)
  num_nodes = len(positions)
  senders = np.repeat(range(num_nodes), [len(a) for a in receivers_list])
  receivers = np.concatenate(receivers_list, axis=0)

  if not add_self_edges:
    # Remove self edges.
    mask = senders != receivers
    senders = senders[mask]
    receivers = receivers[mask]

  return senders, receivers
```

그 다음  $\mathbf x_i$ , $\mathbf r_{i,j}$ 를 이용하여 $\mathbf v_i, \mathbf e_{i,j}$ 그래프를 구성하고, 각 feature 에 대해서는 normalization 을 수행합니다. 수식과 대략적인 개념은 아래와 같습니다.

$$
\mathbf v_i = \varepsilon^v(\mathbf x_i), \quad \mathbf e_{i,j} = \varepsilon^e(\mathbf r_{i,j})
$$

<center>
<figure style="width: 60%"> <img src="/Images/Study/learning_to_simulate/in-graph.jpg" alt="Graph Example"/>
<figcaption>Data Input to Encoder Graph</figcaption>
</figure>
</center>

# Processor
Processor 부분의 네트워크는 노드와 엣지의 네트워크를 Interaction Network (Battaglia et al., 2016) 로 구성하고, `_num_message_passing_steps` 만큼의 네트워크를 구축합니다.

# Decoder
Decoder 에서는 encoder 와는 반대로, 그래프로부터 $\mathbf x_i^{t_{k+1}}$ 를 출력하여, 결과적으로 다음 스텝에서의 입자들의 위치를 예측하는 모델로써 작동합니다. 논문에서는 이러한 네트워크에서, `주변입자탐색` 부분이 CPU에서 수행되고, 전체 플로우에서 큰 부분을 차지하여, 성능이 떨어진다고 언급했는데, 실제로 Water-3D 케이스의 시뮬레이션 1 iteration에 Simulator는 0.104s, GNS의 경우 0.358s 로, Simulator 대비 345% 의 속도를 얻었다고 합니다. 여기서 주변입자 탐색을 제외한 Simulation에 수행된 시간은 0.071s로 19.8% 로, 이는 주변입자탐색 알고리즘을 고도화/최적화 하면 더욱 빨라질 가능성이 있음을 의미합니다.

# References
1. SCARSELLI, Franco, et al. The graph neural network model. IEEE transactions on neural networks, 2008, 20.1: 61-80.
2. SANCHEZ-GONZALEZ, Alvaro, et al. Learning to simulate complex physics with graph networks. In: International Conference on Machine Learning. PMLR, 2020. p. 8459-8468.
3. BATTAGLIA, Peter W., et al. Relational inductive biases, deep learning, and graph networks. arXiv preprint arXiv:1806.01261, 2018.
4. BATTAGLIA, Peter, et al. Interaction networks for learning about objects, relations and physics. Advances in neural information processing systems, 2016, 29.