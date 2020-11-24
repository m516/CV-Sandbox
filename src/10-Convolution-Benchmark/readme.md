# Basic Matrix Convolution Implementation Suite
## *With Benchmarking Tools*

<html>

<body class="c54">
    <p class="c61 title" id="h.mu2hd1tur5z7"><span class="c28"></span></p>
    <hr style="page-break-before:always;display:none;">
    <h1 class="c24 c44" id="h.bupw036ceigp"><span class="c21"></span></h1>
    <h1 class="c24" id="h.5zdapbvrjbm7"><span class="c21">Table of Contents</span></h1>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c55"><span class="c1"><a class="c9" href="#h.5zdapbvrjbm7">Table of Contents</a></span><span class="c1">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c1"><a class="c9" href="#h.5zdapbvrjbm7">2</a></span></p>
    <p class="c36"><span class="c1"><a class="c9" href="#h.q4lrnt7niard">Matrix Convolution</a></span><span class="c1">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c1"><a class="c9" href="#h.q4lrnt7niard">3</a></span></p>
    <p class="c11"><span class="c5"><a class="c9" href="#h.1hobezq1e58h">Introduction</a></span><span class="c5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c5"><a class="c9" href="#h.1hobezq1e58h">3</a></span></p>
    <p class="c11"><span class="c5"><a class="c9" href="#h.dyva3jnek9jl">Aplications</a></span><span class="c5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c5"><a class="c9" href="#h.dyva3jnek9jl">4</a></span></p>
    <p class="c8"><span class="c5"><a class="c9" href="#h.bdtdpoqxsakq">Image Processing</a></span><span class="c5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c5"><a class="c9" href="#h.bdtdpoqxsakq">4</a></span></p>
    <p class="c8"><span class="c5"><a class="c9" href="#h.9g7va96ol1r2">Artificial Intelligence</a></span><span class="c5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c5"><a class="c9" href="#h.9g7va96ol1r2">4</a></span></p>
    <p class="c36"><span class="c1"><a class="c9" href="#h.e7jo3jh5fsz2">Benchmarking</a></span><span class="c1">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c1"><a class="c9" href="#h.e7jo3jh5fsz2">5</a></span></p>
    <p class="c11"><span class="c5"><a class="c9" href="#h.8dxn255cepys">Algorithm</a></span><span class="c5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c5"><a class="c9" href="#h.8dxn255cepys">5</a></span></p>
    <p class="c8"><span class="c5"><a class="c9" href="#h.6z93ofy1b34n">Pseudocode</a></span><span class="c5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c5"><a class="c9" href="#h.6z93ofy1b34n">7</a></span></p>
    <p class="c11"><span class="c5"><a class="c9" href="#h.dcjb9i4io3xc">Benchmarking Algorithms</a></span><span class="c5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c5"><a class="c9" href="#h.dcjb9i4io3xc">8</a></span></p>
    <p class="c8"><span class="c5"><a class="c9" href="#h.k8yhbjpa49j4">void conv4d_convolve_serial_naive();</a></span><span class="c5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c5"><a class="c9" href="#h.k8yhbjpa49j4">8</a></span></p>
    <p class="c8"><span class="c5"><a class="c9" href="#h.z76daj90txak">void conv4d_convolve_serial_discrete();</a></span><span class="c5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c5"><a class="c9" href="#h.z76daj90txak">8</a></span></p>
    <p class="c8"><span class="c5"><a class="c9" href="#h.77u5ef53wfwd">void conv4d_convolve_serial_tiled(int block_size);</a></span><span class="c5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c5"><a class="c9" href="#h.77u5ef53wfwd">8</a></span></p>
    <p class="c8"><span class="c5"><a class="c9" href="#h.9zcv951a9p1s">void conv4d_convolve_threads_discrete();</a></span><span class="c5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c5"><a class="c9" href="#h.9zcv951a9p1s">8</a></span></p>
    <p class="c8"><span class="c5"><a class="c9" href="#h.4j61u8gh8hkx">void conv4d_convolve_threads_tiled(int block_size);</a></span><span class="c5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c5"><a class="c9" href="#h.4j61u8gh8hkx">8</a></span></p>
    <p class="c8"><span class="c5"><a class="c9" href="#h.cdxmt6hvoepr">void conv4d_convolve_OpenMP_discrete();</a></span><span class="c5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c5"><a class="c9" href="#h.cdxmt6hvoepr">9</a></span></p>
    <p class="c8"><span class="c5"><a class="c9" href="#h.wx6oy1ebxf6s">void conv4d_convolve_OpenMP_tiled(int block_size);</a></span><span class="c5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c5"><a class="c9" href="#h.wx6oy1ebxf6s">9</a></span></p>
    <p class="c8"><span class="c5"><a class="c9" href="#h.g1t2f0w3r93b">void conv4d_convolve_CUDA_discrete(int block_size, int grid_size);</a></span><span class="c5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c5"><a class="c9" href="#h.g1t2f0w3r93b">9</a></span></p>
    <p class="c8"><span class="c5"><a class="c9" href="#h.z65tf1qarxn4">void conv4d_convolve_CUDA_discrete_rewrite_gpu_data(int block_size, int grid_size);</a></span><span class="c5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c5"><a class="c9" href="#h.z65tf1qarxn4">9</a></span></p>
    <p class="c11"><span class="c5"><a class="c9" href="#h.3y3x8ri8ai58">Benchmarking Framework</a></span><span class="c5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c5"><a class="c9" href="#h.3y3x8ri8ai58">10</a></span></p>
    <p class="c8"><span class="c5"><a class="c9" href="#h.1cz86shrj6ef">File Structure</a></span><span class="c5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c5"><a class="c9" href="#h.1cz86shrj6ef">10</a></span></p>
    <p class="c8"><span class="c5"><a class="c9" href="#h.pveokbc1p9g7">Data Structures</a></span><span class="c5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c5"><a class="c9" href="#h.pveokbc1p9g7">10</a></span></p>
    <p class="c11"><span class="c5"><a class="c9" href="#h.gwrxwhzao37e">Benchmarking Build Procedure</a></span><span class="c5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c5"><a class="c9" href="#h.gwrxwhzao37e">10</a></span></p>
    <p class="c8"><span class="c5"><a class="c9" href="#h.2etjkb51drtt">Process</a></span><span class="c5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c5"><a class="c9" href="#h.2etjkb51drtt">10</a></span></p>
    <p class="c8"><span class="c5"><a class="c9" href="#h.kp6eim4bxlwy">Hardware Used for Benchmarking</a></span><span class="c5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c5"><a class="c9" href="#h.kp6eim4bxlwy">11</a></span></p>
    <p class="c8"><span class="c5"><a class="c9" href="#h.1905j0wmdn7r">Software Used for Benchmarking</a></span><span class="c5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c5"><a class="c9" href="#h.1905j0wmdn7r">11</a></span></p>
    <p class="c36"><span class="c1"><a class="c9" href="#h.w9dzvh5icfsx">Results</a></span><span class="c1">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c1"><a class="c9" href="#h.w9dzvh5icfsx">12</a></span></p>
    <p class="c11"><span class="c5"><a class="c9" href="#h.z2trwys013cx">TinyImageNet Neural Network</a></span><span class="c5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c5"><a class="c9" href="#h.z2trwys013cx">12</a></span></p>
    <p class="c8"><span class="c5"><a class="c9" href="#h.4wyejyrvqxgv">Tiled Performance</a></span><span class="c5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c5"><a class="c9" href="#h.4wyejyrvqxgv">13</a></span></p>
    <p class="c8"><span class="c5"><a class="c9" href="#h.uwrlf6wchuh">OpenMP Performance</a></span><span class="c5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c5"><a class="c9" href="#h.uwrlf6wchuh">14</a></span></p>
    <p class="c8"><span class="c5"><a class="c9" href="#h.yn4qg6nxutok">CUDA Performance</a></span><span class="c5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c5"><a class="c9" href="#h.yn4qg6nxutok">14</a></span></p>
    <p class="c8"><span class="c5"><a class="c9" href="#h.gijggkhbcawt">Profiling</a></span><span class="c5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c5"><a class="c9" href="#h.gijggkhbcawt">15</a></span></p>
    <p class="c36"><span class="c1"><a class="c9" href="#h.sdtai83rcog7">Conclusions</a></span><span class="c1">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c1"><a class="c9" href="#h.sdtai83rcog7">15</a></span></p>
    <p class="c11"><span class="c5"><a class="c9" href="#h.6bhcidsqxw8o">Toeplitz Matrices</a></span><span class="c5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c5"><a class="c9" href="#h.6bhcidsqxw8o">15</a></span></p>
    <p class="c57"><span class="c5"><a class="c9" href="#h.ihxwepfbjnk4">Fourier Transform</a></span><span class="c5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c5"><a class="c9" href="#h.ihxwepfbjnk4">16</a></span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <hr style="page-break-before:always;display:none;">
    <p class="c4 c6"><span class="c5"></span></p>
    <h1 class="c24" id="h.q4lrnt7niard"><span class="c21">Matrix Convolution</span></h1>
    <h2 class="c30" id="h.1hobezq1e58h"><span class="c19">Introduction</span></h2>
    <p class="c4"><span>Convolution is an operation involving two matrices: the </span><span class="c18">image</span><span>&nbsp;and the </span><span class="c18">kernel</span><span>. During this operation, elements in the </span><span class="c18">convolved</span><span class="c5">&nbsp;matrix are the sum of all nearby values in the image matrix weighted by the kernel. Usually, the kernel matrix is significantly smaller than the image.</span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4"><span class="c5">To demonstrate the process of convolution, I&rsquo;ll convolve two small, 1-dimensional matrices in the following figures. The input matrix (green) is [3,1,4,1,5,9,2], and the filter (red) [1,2,-1].</span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4"><span class="c5">To get the first output of the matrix, add all the surrounding elements in the corresponding input array, weighted by the kernel. </span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4"><span>Element 1: </span><img src="images/image1.png"></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4"><span class="c5">Note that the output matrix must be shifted to the right, so its corresponding element in the input matrix has neighbors in both directions.</span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c46"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 352.00px; height: 352.00px;"><img alt="" src="images/image15.png" style="width: 352.00px; height: 352.00px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p>
    <p class="c4"><span class="c5">The process is repeated for each element in the output matrix by sliding the filter over the input image.</span></p>
    <p class="c4"><span>Element 2: </span><img src="images/image2.png"></p>
    <p class="c4"><span>Element 3: </span><img src="images/image3.png"></p>
    <p class="c4"><span>Element 4: </span><img src="images/image4.png"></p>
    <p class="c4"><span>Element 5: </span><img src="images/image5.png"></p>
    <p class="c46"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 307.50px; height: 301.56px;"><img alt="" src="images/image17.png" style="width: 307.50px; height: 301.56px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 309.72px; height: 307.78px;"><img alt="" src="images/image8.png" style="width: 309.72px; height: 307.78px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p>
    <h2 class="c30" id="h.dyva3jnek9jl"><span class="c19">Aplications</span></h2>
    <p class="c4"><span class="c5">Two common uses for matrix convolution are in image processing and artificial intelligence. Both of these applications are described in the sections below:</span></p>
    <h3 class="c25" id="h.bdtdpoqxsakq"><span class="c7">Image Processing</span></h3>
    <p class="c4"><span class="c5">Many common image filters use matrix convolution. They do so by converting the image into a 3D matrix (width, height, channels, explained in greater detail in the next section)</span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4"><span>A small filter (also known as a </span><span class="c18">kernel</span><span class="c5">) is slid across an image to produce an output image. Cleverly constructed filters can blur and sharpen the image, find corners and edges, and more.</span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <h3 class="c25" id="h.9g7va96ol1r2"><span class="c7">Artificial Intelligence</span></h3>
    <p class="c4"><span>Matrix convolution allows efficient implementations of neural network approximations. Each neuron is an element in a matrix. Convolving the matrix simulates the synapses between neurons. The filter represents the sensitivity of each simulated synapse. By adjusting the values in the filter, each synapse&rsquo;s sensitivity changes, affecting the output from the neural network. Software like TensorFlow contain programs that update these filters, a process known as </span><span class="c18">training</span><span class="c5">.</span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <hr style="page-break-before:always;display:none;">
    <h1 class="c24 c44" id="h.eq2mw8u8ynis"><span class="c21"></span></h1>
    <h1 class="c24" id="h.e7jo3jh5fsz2"><span class="c21">Benchmarking</span></h1>
    <h2 class="c30" id="h.8dxn255cepys"><span class="c19">Algorithm</span></h2>
    <p class="c4"><span class="c5">The process of convolution in this benchmark is an extension of the example in the previous section in a few ways. These extensions were included in the benchmark to increase parallelism complexity for the final project and to support common features in convolutional neural networks. </span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <ul class="c26 lst-kix_n8kr6xnkihiy-0 start">
        <li class="c4 c15"><span class="c5">Convolution occurs in three dimensions. 3D convolution is natural for image processing because each image has three dimensions: width, height, and the number of channels. Figure 6 gives an example of the matrices&rsquo; structures and the variables used in the explanation and implementation of the algorithms. These variables are as follows:</span></li>
    </ul>
    <ul class="c26 lst-kix_n8kr6xnkihiy-1 start">
        <li class="c4 c23"><span class="c5">W, H: The input image&rsquo;s width and height</span></li>
        <li class="c4 c23"><span class="c5">C: The number of channels in the input image. For example, many images have three color channels: red, green, and blue. Sometimes, there is an additional channel for transparency or alpha.</span></li>
        <li class="c4 c23"><span class="c5">R, S: The filter&rsquo;s width and height</span></li>
        <li class="c4 c23"><span class="c5">P, Q: The output&rsquo;s width and height</span></li>
        <li class="c4 c23"><span class="c5">M: The number of channels in the output.</span></li>
        <li class="c4 c23"><span class="c5">N (not shown): The number of input images that need to be applied by the filter. Assumed to be 1 in the diagram.</span></li>
        <li class="c4 c23"><span class="c5">U (not shown): The stride length, i.e. how much the kernel slides across the input matrix. In figures 1-3, the stride length is 1 because the filter shifts to the right by one element each time an element in the output is calculated.</span></li>
    </ul>
    <p class="c46"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 625.84px; height: 332.66px;"><img alt="" src="images/image14.png" style="width: 625.84px; height: 332.66px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p>
    <p class="c4 c47"><span class="c5">The diagram below highlights the cells in each matrix that are used during the first operation.</span></p>
    <p class="c46"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 635.00px; height: 346.26px;"><img alt="" src="images/image18.png" style="width: 635.00px; height: 346.26px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p>
    <ul class="c26 lst-kix_n8kr6xnkihiy-0">
        <li class="c4 c15"><span class="c5">At the end of the convolution operation, the result of the convolution operation is incremented by a vector, called &ldquo;bias.&rdquo; This vector is used predominantly in convolutional neural networks as another way to fine-tune them.</span></li>
        <li class="c4 c15"><span>After the bias was added, a function is applied to each element in the output matrix. This function is called the Rectified Linear Unit, and its function is </span><img src="images/image6.png"><span>&nbsp;if x&lt;0 and </span><img src="images/image7.png"><span class="c5">&nbsp;when x&ge;0.</span></li>
    </ul>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4"><span class="c5">Because this convolution algorithm is targeted towards convolutional neural networks, terminology related to neural networks will be used in the program and explanation. For example, &ldquo;input matrix&rdquo; becomes &ldquo;input feature map,&rdquo; &ldquo;filter&rdquo; or &ldquo;kernel&rdquo; can become &ldquo;weights,&rdquo; etc.</span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <hr style="page-break-before:always;display:none;">
    <p class="c4 c6"><span class="c5"></span></p>
    <h3 class="c25" id="h.6z93ofy1b34n"><span class="c7">Pseudocode</span></h3>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4"><span class="c35">convolve(I, W, B, O):</span></p>
    <p class="c4 c47"><span class="c33 c18">Parameters:</span></p>
    <p class="c4 c17"><span class="c33 c18">I: the input feature map, a four dimensional array with dimensions [N][S+Q*U][R+S*U][C]</span></p>
    <p class="c4 c17"><span class="c18 c33">W: the weights, a four dimensional array with dimensions [S][R][C][M]</span></p>
    <p class="c4 c17"><span class="c33 c18">B: the bias, a one dimensional array with dimension [M]</span></p>
    <p class="c4 c47"><span class="c33 c18">Returns:</span></p>
    <p class="c4 c17"><span class="c18">O: the output feature map, a four dimensional array with dimensions [N][Q][P][M]</span><span><br></span><span class="c1">Reset O so that each element is 0</span></p>
    <p class="c4 c47"><span class="c35">For each </span><span class="c35 c18">n</span><span class="c1">&nbsp;in N:</span></p>
    <p class="c4 c47"><span class="c35">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For each </span><span class="c35 c18">q</span><span class="c1">&nbsp;in Q:</span></p>
    <p class="c4 c47"><span class="c35">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For each </span><span class="c35 c18">p</span><span class="c1">&nbsp;in P:</span></p>
    <p class="c4 c47"><span class="c35">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For each </span><span class="c18 c35">m</span><span class="c1">&nbsp;in M:</span></p>
    <p class="c4 c47"><span class="c35">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For each </span><span class="c35 c18">s</span><span class="c1">&nbsp;in S:</span></p>
    <p class="c4 c47"><span class="c35">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For each </span><span class="c35 c18">r</span><span class="c1">&nbsp;in R:</span></p>
    <p class="c4 c47"><span class="c35">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For each </span><span class="c35 c18">c</span><span class="c1">&nbsp;in C:</span></p>
    <p class="c4 c47"><span class="c1">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Increase O[n][q][p][m] by </span></p>
    <p class="c4 c58"><span class="c1">I[n][q*U+s][p*U+r] * W[s][r][c][m]</span></p>
    <p class="c4 c48"><span class="c35">Increase O[n][q][p][m] by B[m]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c33 c18">Bias</span></p>
    <p class="c4 c48"><span class="c35">O[n][q][p][m] = f(O[n][q][p][m]) </span><span class="c33 c18">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Apply the activation function</span></p>
    <p class="c4 c6 c48"><span class="c1"></span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <hr style="page-break-before:always;display:none;">
    <p class="c4 c6"><span class="c5"></span></p>
    <h2 class="c30" id="h.dcjb9i4io3xc"><span class="c19">Benchmarking Algorithms</span></h2>
    <p class="c4"><span class="c5">The benchmarking framework contains nine variations of the convolution algorithm described in the previous section, described below:</span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <h3 class="c10" id="h.k8yhbjpa49j4"><span class="c22">void</span><span class="c14">&nbsp;</span><span class="c43">conv4d_convolve_serial_naive</span><span class="c14 c31">();</span></h3>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4"><span class="c5">This algorithm convolves the input feature map by following the pseudocode verbatim, ignoring the machine&rsquo;s hardware.</span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <h3 class="c10" id="h.z76daj90txak"><span class="c22">void</span><span class="c14">&nbsp;</span><span class="c43">conv4d_convolve_serial_discrete</span><span class="c14 c31">();</span></h3>
    <p class="c4 c6"><span class="c3"></span></p>
    <p class="c4"><span>This algorithm improves in performance because it splits the loops for convolution and bias, giving the compiler more flexibility to perform its assembly-level optimizations.</span></p>
    <p class="c4 c6"><span class="c3"></span></p>
    <h3 class="c10" id="h.77u5ef53wfwd"><span class="c22">void</span><span class="c14">&nbsp;</span><span class="c43">conv4d_convolve_serial_tiled</span><span class="c14">(</span><span class="c22">int</span><span class="c14">&nbsp;</span><span class="c41">block_size</span><span class="c14 c31">);</span></h3>
    <p class="c4 c6"><span class="c3"></span></p>
    <p class="c4"><span>This algorithm attempts to further improve the performance by tiling the arrays, theoretically increasing cache hits.</span></p>
    <p class="c4 c6"><span class="c3"></span></p>
    <h3 class="c10" id="h.9zcv951a9p1s"><span class="c22">void</span><span class="c14">&nbsp;</span><span class="c43">conv4d_convolve_threads_discrete</span><span class="c14 c31">();</span></h3>
    <p class="c4 c6"><span class="c3"></span></p>
    <p class="c4"><span>This algorithm is similar to conv4d_convolve_serial_discrete, except that the output feature map&rsquo;s rows are distributed among threads using the Pthread library. </span></p>
    <p class="c4 c6"><span class="c3"></span></p>
    <h3 class="c10" id="h.4j61u8gh8hkx"><span class="c22">void</span><span class="c14">&nbsp;</span><span class="c43">conv4d_convolve_threads_tiled</span><span class="c14">(</span><span class="c22">int</span><span class="c14">&nbsp;</span><span class="c41">block_size</span><span class="c14 c31">);</span></h3>
    <p class="c4 c6"><span class="c3"></span></p>
    <p class="c4"><span>This algorithm is similar to conv4d_convolve_serial_tiled, except that the output feature map&rsquo;s rows are distributed among threads using the Pthread library. </span></p>
    <p class="c4 c6"><span class="c3"></span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4 c6"><span class="c3"></span></p>
    <h3 class="c10" id="h.cdxmt6hvoepr"><span class="c22">void</span><span class="c14">&nbsp;</span><span class="c43">conv4d_convolve_OpenMP_discrete</span><span class="c14 c31">();</span></h3>
    <p class="c4 c6"><span class="c3"></span></p>
    <p class="c4"><span class="c5">This algorithm is similar to conv4d_convolve_serial_naive, except that it uses OpenMP to distribute the workload between threads. Each element in the output feature map is its own task and is run in parallel with other elements.</span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4"><span class="c5">It was based off of the naive version because the discrete version created a race condition.</span></p>
    <p class="c4 c6"><span class="c3"></span></p>
    <h3 class="c10" id="h.wx6oy1ebxf6s"><span class="c29">void</span><span class="c40">&nbsp;</span><span class="c56">conv4d_convolve_OpenMP_tiled</span><span class="c40">(</span><span class="c29">int</span><span class="c40">&nbsp;</span><span class="c32">block_size</span><span class="c40 c31">);</span></h3>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4"><span class="c5">This algorithm is similar to conv4d_convolve_serial_tiled, except that it uses OpenMP to distribute the workload between threads. Each element in the output feature map is its own task and is run in parallel with other elements.</span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4"><span>However, instead of tiling over </span><span class="c18">q</span><span>&nbsp;and </span><span class="c18">p</span><span>, tiling is performed over </span><span class="c18">c</span><span>&nbsp;so that the first three </span><span class="c18">for</span><span class="c5">&nbsp;loops can be collapsed. Again, this also helps remove the race condition, avoiding the need of a critical region.</span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <h3 class="c10" id="h.g1t2f0w3r93b"><span class="c22">void</span><span class="c14">&nbsp;</span><span class="c43">conv4d_convolve_CUDA_discrete</span><span class="c14">(</span><span class="c22">int</span><span class="c14">&nbsp;</span><span class="c41">block_size</span><span class="c14">, </span><span class="c22">int</span><span class="c14">&nbsp;</span><span class="c41">grid_size</span><span class="c14 c31">);</span></h3>
    <p class="c4 c6"><span class="c3"></span></p>
    <p class="c4"><span>This algorithm is similar to conv4d_convolve_serial_discrete, except that it uses CUDA to distribute the workload between threads on the GPU. Each element in the output feature map is its own task and is run in parallel with other elements. Blocking happens naturally due to Nvidia GPUs&rsquo; architectures.</span></p>
    <p class="c4 c6"><span class="c3"></span></p>
    <h3 class="c10" id="h.z65tf1qarxn4"><span class="c22">void</span><span class="c14">&nbsp;</span><span class="c43">conv4d_convolve_CUDA_discrete_rewrite_gpu_data</span><span class="c14">(</span><span class="c22">int</span><span class="c14">&nbsp;</span><span class="c41">block_size</span><span class="c14">, </span><span class="c22">int</span><span class="c14">&nbsp;</span><span class="c41">grid_size</span><span class="c14 c31">);</span></h3>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4"><span class="c5">This algorithm is conv4d_convolve_CUDA_discrete, except that its memory policy is a bit different.</span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4"><span class="c5">The CUDA functions use &nbsp;their own global, device-specific memory. To convolve a matrix in CPU memory with CUDA, this program copies that matrix and the filter over to the device memory. Then a GPU kernel can run on the GPU memory. Finally, after the kernel completes, the output matrix (in GPU memory) is copied back to the CPU.</span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4"><span>This function copies the input feature map and layer from the CPU memory into the GPU before running conv4d_convolve_CUDA_discrete, costing time.</span></p>
    <p class="c4 c6"><span class="c31 c59"></span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <h2 class="c30" id="h.3y3x8ri8ai58"><span class="c19">Benchmarking Framework</span></h2>
    <p class="c4"><span class="c5">In addition to all nine functions, the benchmarking framework also contains tools to profile each algorithm and verify its correctness.</span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <h3 class="c25" id="h.1cz86shrj6ef"><span class="c7">File Structure</span></h3>
    <p class="c4"><span class="c5">The project contains seven files:</span></p>
    <ul class="c26 lst-kix_9wdgkqjvcy5b-0 start">
        <li class="c4 c15"><span class="c5">CMakeLists.txt: A file used by CMake to build the project. It automatically disables benchmarks that aren&rsquo;t supported by the machine.</span></li>
        <li class="c4 c15"><span class="c5">conv4D_data_structures.h: Definitions of feature maps and functions that aid in the benchmarking process.</span></li>
        <li class="c4 c15"><span class="c5">conv4D_data_structures.c: Implementation for functions in conv4D_data_structures.h.</span></li>
        <li class="c4 c15"><span class="c5">conv4D_impl.h: Definitions of the benchmarking algorithms in the previous section.</span></li>
        <li class="c4 c15"><span class="c5">conv4D_impl_CPU.c: Implementations of the benchmark that predominantly use the CPU (serial, OpenMP, and threads)</span></li>
        <li class="c4 c15"><span class="c5">conv4D_impl_GPU.cu: Implementations of the benchmark that predominantly use the GPU (CUDA)</span></li>
        <li class="c4 c15"><span class="c5">main.c: the main() function that benchmarks each algorithm with a macro called BENCHMARK_ALGORITHM. The marco times the execution of an algorithm, calculates its average error, and prints the following information on one line, separated by a comma and a tab:</span></li>
    </ul>
    <ul class="c26 lst-kix_9wdgkqjvcy5b-1 start">
        <li class="c4 c23"><span class="c5">Algorithm name</span></li>
        <li class="c4 c23"><span class="c5">Average execution time, not including the first trial. Lower values indicate higher performance.</span></li>
        <li class="c4 c23"><span class="c5">The average difference between the output and the expected output per unit. Lower values indicate higher accuracy.</span></li>
        <li class="c4 c23"><span>Parameters in the function, if any.</span></li>
    </ul>
    <p class="c4"><span>All files are placed in the folder </span><span class="c18">src/10_Convolution_Benchmark</span><span class="c5">. </span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <h3 class="c25" id="h.pveokbc1p9g7"><span class="c7">Data Structures</span></h3>
    <p class="c4"><span>The convolutional layer and each feature map have their own data type and are stored in global memory. They&rsquo;re defined in conv4D_data_structures.h.</span></p>
    <h2 class="c30" id="h.gwrxwhzao37e"><span class="c19">Benchmarking Build Procedure</span></h2>
    <h3 class="c25" id="h.2etjkb51drtt"><span class="c7">Process</span></h3>
    <p class="c4"><span>Step 1: Download a copy of the full project from the GitHub repository: </span><span class="c38"><a class="c9" href="https://www.google.com/url?q=https://github.com/m516/CV-Sandbox&amp;sa=D&amp;ust=1606191316203000&amp;usg=AOvVaw3a2pr48f6s7YckpR7IGOTj">https://github.com/m516/CV-Sandbox</a></span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4"><span class="c5">Step 2: In the root directory of the repository, run the following command:</span></p>
    <p class="c4"><span class="c53">cmake .</span></p>
    <p class="c4"><span class="c5">Step 3: Build the tool with make if you&rsquo;re using Linux or Mac:</span></p>
    <p class="c4"><span class="c53">$ make 10_CONVOLUTION_BENCHMARK</span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4"><span class="c5">Step 4: Run the program (contained in the bin subdirectory of the project folder)</span></p>
    <p class="c4"><span class="c53">$ bin/10_CONVOLUTION_BENCHMARK</span></p>
    <h3 class="c25" id="h.kp6eim4bxlwy"><span class="c7">Hardware Used for Benchmarking</span></h3>
    <p class="c4"><span class="c5">I used a Lenovo Yoga 730 to perform benchmarking. It has the following hardware components:</span></p>
    <ul class="c26 lst-kix_53pypmvu12q7-0 start">
        <li class="c4 c15"><span class="c5">CPU: Intel Core i7-8550U</span></li>
    </ul>
    <ul class="c26 lst-kix_53pypmvu12q7-1 start">
        <li class="c4 c23"><span class="c5">Base clock: 1.80GHz</span></li>
        <li class="c4 c23"><span class="c5">Max. clock: 4 GHz</span></li>
        <li class="c4 c23"><span class="c5">Cache:</span></li>
    </ul>
    <ul class="c26 lst-kix_53pypmvu12q7-2 start">
        <li class="c4 c49"><span class="c5">L1 data: 128 kB</span></li>
        <li class="c4 c49"><span class="c5">L1 instruction: 128 kB</span></li>
        <li class="c4 c49"><span class="c5">L2: 1 MB</span></li>
        <li class="c4 c49"><span class="c5">L3: 8 MB</span></li>
    </ul>
    <ul class="c26 lst-kix_53pypmvu12q7-1">
        <li class="c4 c23"><span class="c5">Cores: 4</span></li>
        <li class="c4 c23"><span class="c5">Threads: 8</span></li>
        <li class="c4 c23"><span class="c5">NUMA nodes: 8</span></li>
    </ul>
    <ul class="c26 lst-kix_53pypmvu12q7-0">
        <li class="c4 c15"><span class="c5">RAM: 16 GB DDR4, 2400MHz</span></li>
        <li class="c4 c15"><span class="c5">GPU: Nvidia GTX 1050 Mobile</span></li>
    </ul>
    <ul class="c26 lst-kix_53pypmvu12q7-1 start">
        <li class="c4 c23"><span class="c5">Clock: 33MHz</span></li>
        <li class="c4 c23"><span class="c5">4 GB dedicated RAM</span></li>
    </ul>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4"><span class="c5">The benchmark attempts to use as many resources as possible. OpenMP and Pthread implementations both take all 8 hardware threads.</span></p>
    <h3 class="c25" id="h.1905j0wmdn7r"><span class="c7">Software Used for Benchmarking</span></h3>
    <ul class="c26 lst-kix_22jkesalxpd2-0 start">
        <li class="c4 c15"><span class="c5">Operating system: agnostic (benchmarked with Ubuntu 20.04)</span></li>
        <li class="c4 c15"><span class="c5">Language: C</span></li>
        <li class="c4 c15"><span class="c5">Compiler: agnostic (benchmarked with GCC 9.3.0) </span></li>
        <li class="c4 c15"><span class="c5">Build tool: CMake</span></li>
        <li class="c4 c15"><span class="c5">Additional packages and libraries:</span></li>
    </ul>
    <ul class="c26 lst-kix_22jkesalxpd2-1 start">
        <li class="c4 c23"><span class="c5">OpenMP</span></li>
        <li class="c4 c23"><span class="c5">CUDA</span></li>
        <li class="c4 c23"><span class="c5">PThread</span></li>
    </ul>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <h1 class="c24" id="h.w9dzvh5icfsx"><span class="c21">Results</span></h1>
    <h2 class="c30" id="h.z2trwys013cx"><span class="c19">TinyImageNet Neural Network</span></h2>
    <p class="c4"><span class="c5">The first benchmark is based on intermediate feature maps that were extracted from a TinyImageNet neural network running on Tensorflow. The extracted feature map and convolutional layer data were placed in binary files under the media/dnn directory in the project folder. To ensure that all algorithms work as expected, the output feature map was also extracted into a file, and the values in the calculated feature map are compared with the corresponding values in the file.</span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4"><span class="c5">These are the dimensions of the feature map:</span></p>
    <ul class="c26 lst-kix_4saw2tkv56b9-0 start">
        <li class="c4 c15"><span class="c5">Input feature map filename: &quot;dnn/Test_Input0/layer_0_output.bin&quot;</span></li>
        <li class="c4 c15"><span class="c5">Layer weights filename: &quot;dnn/Test_Input0/conv2_weights.bin&quot;</span></li>
        <li class="c4 c15"><span class="c5">Layer bias filename: &quot;dnn/Test_Input0/conv2_biases.bin&quot;</span></li>
        <li class="c4 c15"><span class="c5">Input feature map filename: &quot;dnn/Test_Input0/layer_1_output.bin&quot;</span></li>
        <li class="c4 c15"><span class="c5">Batch size: 1</span></li>
        <li class="c4 c15"><span class="c5">Input feature map: 60x60x32</span></li>
        <li class="c4 c15"><span class="c5">Layer: 5x5x32</span></li>
        <li class="c4 c15"><span class="c5">Stride length: 1</span></li>
    </ul>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4"><span class="c5">Here is the time required to compute each algorithm with this data (highest performing algorithm in bold):</span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <a id="t.8b21416fca8abfc99c334d66c160b132cdcba1f9"></a>
    <a id="t.0"></a>
    <table class="c34">
        <tbody>
            <tr class="c50">
                <td class="c45" colspan="1" rowspan="1">
                    <p class="c12"><span class="c13">Algorithm</span></p>
                </td>
                <td class="c51" colspan="1" rowspan="1">
                    <p class="c12"><span class="c13">Average Execution Time (s)</span></p>
                </td>
                <td class="c16" colspan="1" rowspan="1">
                    <p class="c12"><span class="c13">Error per Element in Output Feature Map</span></p>
                </td>
                <td class="c37" colspan="1" rowspan="1">
                    <p class="c12"><span class="c13">Speedup from serial_naive</span></p>
                </td>
                <td class="c27" colspan="1" rowspan="1">
                    <p class="c12"><span class="c13">Speedup from serial_discrete</span></p>
                </td>
                <td class="c20" colspan="1" rowspan="1">
                    <p class="c12"><span class="c13">GFLOPS</span></p>
                </td>
            </tr>
            <tr class="c2">
                <td class="c45" colspan="1" rowspan="1">
                    <p class="c12"><span class="c13">serial_naive</span></p>
                </td>
                <td class="c51" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">0.0858</span></p>
                </td>
                <td class="c16" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">0</span></p>
                </td>
                <td class="c37" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">1.00</span></p>
                </td>
                <td class="c27" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">0.09</span></p>
                </td>
                <td class="c20" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">1.87</span></p>
                </td>
            </tr>
            <tr class="c2">
                <td class="c45" colspan="1" rowspan="1">
                    <p class="c12"><span class="c13">serial_discrete</span></p>
                </td>
                <td class="c51" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">0.0080</span></p>
                </td>
                <td class="c16" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">0</span></p>
                </td>
                <td class="c37" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">10.77</span></p>
                </td>
                <td class="c27" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">1.00</span></p>
                </td>
                <td class="c20" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">20.16</span></p>
                </td>
            </tr>
            <tr class="c2">
                <td class="c45" colspan="1" rowspan="1">
                    <p class="c12"><span class="c13">serial_tiled</span></p>
                </td>
                <td class="c51" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">0.0550</span></p>
                </td>
                <td class="c16" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">0</span></p>
                </td>
                <td class="c37" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">1.56</span></p>
                </td>
                <td class="c27" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">0.14</span></p>
                </td>
                <td class="c20" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">2.92</span></p>
                </td>
            </tr>
            <tr class="c2">
                <td class="c45" colspan="1" rowspan="1">
                    <p class="c12"><span class="c13">threads_discrete</span></p>
                </td>
                <td class="c51" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">0.0039</span></p>
                </td>
                <td class="c16" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">0</span></p>
                </td>
                <td class="c37" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">22.19</span></p>
                </td>
                <td class="c27" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">2.06</span></p>
                </td>
                <td class="c20" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">41.52</span></p>
                </td>
            </tr>
            <tr class="c2">
                <td class="c45" colspan="1" rowspan="1">
                    <p class="c12"><span class="c13">threads_tiled</span></p>
                </td>
                <td class="c51" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">0.0035</span></p>
                </td>
                <td class="c16" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">0</span></p>
                </td>
                <td class="c37" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">24.60</span></p>
                </td>
                <td class="c27" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">2.28</span></p>
                </td>
                <td class="c20" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">46.03</span></p>
                </td>
            </tr>
            <tr class="c2">
                <td class="c45" colspan="1" rowspan="1">
                    <p class="c12"><span class="c31 c42 c35">OpenMP_discrete</span></p>
                </td>
                <td class="c51" colspan="1" rowspan="1">
                    <p class="c0"><span class="c31 c35 c42">0.0022</span></p>
                </td>
                <td class="c16" colspan="1" rowspan="1">
                    <p class="c0"><span class="c31 c42 c35">0</span></p>
                </td>
                <td class="c37" colspan="1" rowspan="1">
                    <p class="c0"><span class="c31 c42 c35">38.18</span></p>
                </td>
                <td class="c27" colspan="1" rowspan="1">
                    <p class="c0"><span class="c31 c42 c35">3.54</span></p>
                </td>
                <td class="c20" colspan="1" rowspan="1">
                    <p class="c0"><span class="c31 c42 c35">71.43</span></p>
                </td>
            </tr>
            <tr class="c2">
                <td class="c45" colspan="1" rowspan="1">
                    <p class="c12"><span class="c13">OpenMP_tiled</span></p>
                </td>
                <td class="c51" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">0.0170</span></p>
                </td>
                <td class="c16" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">0</span></p>
                </td>
                <td class="c37" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">5.06</span></p>
                </td>
                <td class="c27" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">0.47</span></p>
                </td>
                <td class="c20" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">9.46</span></p>
                </td>
            </tr>
            <tr class="c2">
                <td class="c45" colspan="1" rowspan="1">
                    <p class="c12"><span class="c13">cuda_discrete</span></p>
                </td>
                <td class="c51" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">0.0369</span></p>
                </td>
                <td class="c16" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">0</span></p>
                </td>
                <td class="c37" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">2.32</span></p>
                </td>
                <td class="c27" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">0.22</span></p>
                </td>
                <td class="c20" colspan="1" rowspan="1">
                    <p class="c0"><span class="c13">4.35</span></p>
                </td>
            </tr>
        </tbody>
    </table>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4"><span class="c5">According to this table, OpenMP and threading both significantly speed up the convolution program when used effectively, but OpenMP was more effective.</span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <h3 class="c25" id="h.4wyejyrvqxgv"><span class="c7">Tiled Performance</span></h3>
    <p class="c4"><span>Most implementations of tiled convolution had worse performance than their discrete counterparts. When GCC optimized the </span><span class="c18">for</span><span class="c5">&nbsp;loops, it tried to distribute the instructions to minimize cache misses. It was able to optimize the chain of for loops very well, and it automatically vectorized operations with SIMD. My tiling interfered with many of those optimizations, reducing the overall performance of the algorithm.</span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4"><span class="c5">In other words, GCC was much better at optimizing the loops for my machine than I was. Boy, I still have a lot to learn!</span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4"><span class="c5">However, tiled convolution performed better than discrete convolution on Pthreads. I believe this is because the DRAM bandwidth imposes a bottleneck on this program, and tiling gave the program the opportunity to use data in the cache multiple times. That would explain the spike that occurs when block size is 6.</span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 333.58px; height: 206.17px;"><img alt="" src="images/image10.png" style="width: 333.58px; height: 206.17px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title="Chart"></span><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 321.35px; height: 198.17px;"><img alt="" src="images/image13.png" style="width: 321.35px; height: 198.17px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title="Chart"></span><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 335.70px; height: 207.48px;"><img alt="" src="images/image12.png" style="width: 335.70px; height: 207.48px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title="Chart"></span></p>
    <h3 class="c25" id="h.uwrlf6wchuh"><span class="c7">OpenMP Performance</span></h3>
    <p class="c4"><span class="c5">Auto, static, and guided scheduling methods all performed about the same on this benchmark, and dynamic scheduling took significantly longer than the other three. That is expected with this algorithm because every task takes almost the same amount of time, since conditional branching is minimal.</span></p>
    <p class="c46"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 497.49px; height: 308.34px;"><img alt="" src="images/image9.png" style="width: 497.49px; height: 308.34px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title="Chart"></span></p>
    <p class="c4"><span class="c5">I was unable to use Allinea or MAP because I don&rsquo;t have the programs on my machine. I was able to profile the CUDA algorithms because the CUDA development kit came with its own tool: nvprof.</span></p>
    <p class="c46 c6"><span class="c5"></span></p>
    <h3 class="c25" id="h.yn4qg6nxutok"><span class="c7">CUDA Performance</span></h3>
    <p class="c4"><span class="c5">CUDA did not perform very well, but that&rsquo;s likely because I don&rsquo;t have much experience in the language or framework. However, the algorithm was parallelizable since latency generally decreased when grid_size and block_size increases.</span></p>
    <p class="c46"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 535.83px; height: 331.29px;"><img alt="" src="images/image11.png" style="width: 535.83px; height: 331.29px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title="Chart"></span></p>
    <h3 class="c25" id="h.gijggkhbcawt"><span class="c7">Profiling</span></h3>
    <p class="c4"><span class="c5">Below is the output from profiling the discrete CUDA algorithm with a block size of 4 and a grid size of 4.</span></p>
    <p class="c4"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 720.00px; height: 342.67px;"><img alt="" src="images/image16.png" style="width: 720.00px; height: 342.67px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4"><span class="c5">According to the tool, synchronizing all the data between the CPU and GPU memory took about 10 ms. Convolution took 120x longer on average than memory synchronization. </span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4"><span>The average throughput was 2.17 FLOPS, and the peak theoretical performance is 1.8 TFLOPS (</span><span class="c38"><a class="c9" href="https://www.google.com/url?q=https://www.techpowerup.com/gpu-specs/geforce-gtx-1050.c2875&amp;sa=D&amp;ust=1606191316227000&amp;usg=AOvVaw08llJ6xb2n2cZRJt7bPWA8">source: TechPowerUp</a></span><span class="c5">). This suggests that my algorithm doesn&rsquo;t efficiently use the hardware resources in the graphics card.</span></p>
    <h1 class="c24" id="h.sdtai83rcog7"><span class="c21">Conclusions</span></h1>
    <p class="c4"><span class="c5">There is much room for improvement in all the algorithms that were benchmarked in this project.</span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <p class="c4"><span class="c5">Memory is a bottleneck for these algorithms because discrete convolution doesn&rsquo;t allow stride-1 array accesses to all matrices. In this project, I haven&rsquo;t covered at least two advanced optimization techniques that can dramatically improve memory access times by maximizing or enforcing stride-1 array accesses and data reuse.</span></p>
    <p class="c4 c6"><span class="c5"></span></p>
    <h2 class="c30" id="h.6bhcidsqxw8o"><span class="c19">Toeplitz Matrices</span></h2>
    <p class="c4"><span>Matrix convolution can be transformed into matrix multiplication by converting one of the matrices into its Toeplitz matrix (</span><span class="c38"><a class="c9" href="https://www.google.com/url?q=https://en.wikipedia.org/wiki/Toeplitz_matrix&amp;sa=D&amp;ust=1606191316228000&amp;usg=AOvVaw2oQ4dnjcrXBRyMgV0oK5j5">source: Wikipedia</a></span><span class="c5">). An additional latency is incurred by the conversion process, but the benefit of enforcing higher amounts of stride-1 array accesses would outweigh the cost of that latency in sufficiently large matrices.</span></p>
    <h2 class="c30" id="h.ihxwepfbjnk4"><span class="c19">Fourier Transform</span></h2>
    <p class="c4"><span>Convolutions of large matrices are usually implemented most quickly using the Fourier transform (</span><span class="c38"><a class="c9" href="https://www.google.com/url?q=https://ccrma.stanford.edu/~jos/filters/Cyclic_Convolution_Matrix.html&amp;sa=D&amp;ust=1606191316228000&amp;usg=AOvVaw22xxg9bJ6oZImB9xWjP5xU">source: ccrma.stanford.edu</a></span><span>) because only element-wise multiplication is necessary in the Fourier transform domain. In other words, it&rsquo;s possible to take the Fourier transform of both, perform element-wise multiplication, then perform the inverse Fourier transform to convolve two matrices (</span><span class="c38"><a class="c9" href="https://www.google.com/url?q=http://pillowlab.princeton.edu/teaching/mathtools16/slides/lec23_Fourier.pdf&amp;sa=D&amp;ust=1606191316229000&amp;usg=AOvVaw1T_12SV_P_OuodIh1Aa5s_">http://pillowlab.princeton.edu</a></span><span class="c5">).</span></p>
</body>

</html>
