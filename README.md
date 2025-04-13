<div align="center">

<h1>In-2-4D: Inbetweening from Two Single-View Images <br> to 4D Generation </h1>

<div>
    <a href='https://sauradip.github.io/' target='_blank'>Sauradip Nag</a><sup>1</sup>&emsp;
  <a href='https://danielcohenor.com/' target='_blank'>Daniel Cohen-Or</a><sup>2</sup>&emsp; 
   <a href='https://www2.cs.sfu.ca/~haoz/index.html' target='_blank'>Richard (Hao) Zhang</a><sup>1</sup>&emsp; 
    <a href='https://www.sfu.ca/~amahdavi' target='_blank'>Ali Mahdavi Amiri</a><sup>1</sup>&emsp;
</div>
<div>
    <sup>1</sup>Simon Fraser University
    <sup>2</sup>Tel Aviv University
        &emsp; <br>
</div>


<h3 align="center">
  <a href="" target='_blank'>Paper</a> |
  <a href="https://in-2-4d.github.io" target='_blank'>Project Page</a> 
</h3>
</div>
We propose a new problem, In-2-4D, for generative 4D (i.e., 3D + motion) inbetweening from a minimalistic input setting: two single-view images capturing an object in two distinct motion states. Given two images representing the start and end states of an object in motion, our goal is to generate and reconstruct the motion in 4D. We utilize a video interpolation model to predict the motion, but large frame-to-frame motions can lead to ambiguous interpretations. To overcome this, we employ a hierarchical approach to identify keyframes that are visually close to the input states and show significant motion, then generate smooth fragments between them. For each fragment, we construct the 3D representation of the keyframe using Gaussian Splatting. The temporal frames within the fragment guide the motion, enabling their transformation into dynamic Gaussians through a deformation field. To improve temporal consistency and refine 3D motion, we expand the self-attention of multi-view diffusion across timesteps and apply rigid transformation regularization. Finally, we merge the independently generated 3D motion segments by interpolating boundary deformation fields and optimizing them to align with the guiding video, ensuring smooth and flicker-free transitions.

<div align="center">
<table>
<tr>
    <td><img src="https://github.com/alimohammadiamirhossein/smite/blob/main/assets/teaser.png" width="100%"/></td>
</tr>
</table>
</div>
