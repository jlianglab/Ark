<p align="center"><img width=15% alt="FrontCover" src="media/Ark_logo.png"></p>

# Foundation Ark: Accruing and Reusing Knowledge

We develop **open Foundation Models** from numerous public datasets using their heterogeneous expert annotations. Our Ark models outperform SOTA fully/self-supervised methods on various thoracic disease classification tasks and organ/bones segmentation tasks. Ark offers embeddings with superior quality over [Google's CXR Foundation Model](https://github.com/Google-Health/imaging-research/tree/master/cxr-foundation).





## Publication
<b>Foundation Ark: Accruing and Reusing Knowledge for Superior and Robust Performance </b> <br/>
[DongAo Ma](https://github.com/Mda233)<sup>1</sup>, [Jiaxuan Pang](https://github.com/MRJasonP)<sup>1</sup>, [Michael B. Gotway](https://www.mayoclinic.org/biographies/gotway-michael-b-m-d/bio-20055566)<sup>2</sup>, [Jianming Liang](https://chs.asu.edu/jianming-liang)<sup>1</sup><br/>
<sup>1 </sup>Arizona State University, <sup>2 </sup>Mayo Clinic <br/>

International Conference on Medical Image Computing and Computer Assisted Intervention ([MICCAI 2023](https://conferences.miccai.org/2023/en/))

[Paper](https://link.springer.com/chapter/10.1007/978-3-031-43907-0_62) | [Code](https://github.com/jlianglab/Ark) | [Poster](media/Ark_poster.pdf) | [Slides]() | Presentation ([YouTube]())


### Requirements
+ Python
+ PyTorch ([pytorch.org](http://pytorch.org))



## Pre-trained models

You can download the pretrained models used/developed in our paper as follows:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Training Dataset</th>
<th valign="bottom">model</th>

<tr>
<td>Ark-5</td>
<td>Swin-Base</td>
<td>ImageNet &#8594; X-rays(335K)</td>
<td><a href="">Preparing</a></td>

</tr>
<tr> 
<td>Ark-6</td>
 <td>Swin-Base</td>
<td>ImageNet &#8594; X-rays(704K)</td>

<td><a href="">Preparing</a></td>
 </tr>

 
</tbody></table>



## Citation
If you use this code or use our pre-trained weights for your research, please cite our paper:
```
@InProceedings{ma2023foundation,
    author="Ma, DongAo and Pang, Jiaxuan and Gotway, Michael B. and Liang, Jianming",
    title="Foundation Ark: Accruing and Reusing Knowledge for Superior and Robust Performance",
    booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023",
    year="2023",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="651--662",
    isbn="978-3-031-43907-0"
}
```

## Acknowledgement
This research has been supported in part by ASU and Mayo Clinic through a Seed Grant and an Innovation Grant, and in part by the NIH under Award Number R01HL128785. The content is solely the responsibility of the authors and does not necessarily represent the official views of the NIH. This work has utilized the GPUs provided in part by the ASU Research Computing and in part by the Bridges-2 at Pittsburgh Supercomputing Center through allocation BCS190015 and the Anvil at Purdue University through allocation MED220025 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296. We also acknowledge Google for granting us access to CXR Foundation API, which enabled us to generate the embeddings for the target datasets. The content of this paper is covered by patents pending.


## License

Released under the [ASU GitHub Project License](./LICENSE).# Ark
