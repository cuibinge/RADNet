# 作者：赵衍利
Reverse Attention Dual-Stream Network for Extracting Laver Aquaculture Areas From GF-1 Remote Sensing Images

Extracting laver aquaculture areas from remote sensing images is very important for laver aquaculture monitoring and scientific management. However, due to the large differences in spectral features of laver aquaculture areas caused by factors such as different growth stages and harvesting conditions, traditional machine learning and deep learning methods face great challenges in achieving accurate and complete extraction of raft laver aquaculture areas. In this article, a reverse attention dual-stream network (RADNet) is proposed for the extraction of laver aquaculture areas with weak spectral responses by comprehensively considering both the aquaculture boundary and surrounding sea background information. RADNet consists of a boundary stream and a segmentation stream. Considering the weaker spectral responses of certain laver aquaculture areas, we introduce a reverse attention module in the segmentation stream to amplify the weaker responses of inapparent laver aquaculture areas. To suppress the response of nonboundary details in the boundary stream, we design a boundary attention module, which is guided by high-level semantics from the segmentation stream. The structural information of the laver aquaculture area learned from the boundary stream will be fed back to the segmentation stream through a specially designed boundary guidance module. The study is conducted in Haizhou Bay, China, and is verified using a self-labeled GF-1 multispectral dataset. The experimental results show that RADNet model performs better in extracting inapparent laver aquaculture areas compared to SOTA models.

基于反向注意双流网络从 GF-1 遥感图像中提取紫菜养殖区域

从遥感影像中提取紫菜养殖区域对紫菜养殖监测和科学管理具有重要意义。然而由于不同生长阶段、采收条件等因素造成紫菜养殖区域光谱特征差异较大，传统的机器学习和深度学习方法在实现筏式紫菜养殖区域的准确完整提取方面面临巨大挑战。本文提出了一种反向注意双流网络（RADNet），用于综合考虑养殖边界和周围海域背景信息，提取光谱响应较弱的紫菜养殖区域。RADNet由边界流和分割流组成。考虑到某些紫菜养殖区域的光谱响应较弱，我们在分割流中引入反向注意模块，以放大不明显紫菜养殖区域的较弱响应。为了抑制边界流中非边界细节的响应，我们设计了一个边界注意模块，该模块由来自分割流的高级语义引导。从边界流中学习到的紫菜养殖区域结构信息将通过专门设计的边界引导模块反馈到分割流中。研究在中国海州湾进行，并使用自标记的GF-1多光谱数据集进行验证。实验结果表明，RADNet模型在提取不明显的紫菜养殖区域方面比SOTA模型表现更好。
https://ieeexplore.ieee.org/document/10141641

Running environment:

    python == 3.6.16
    keras == 2.1.5
    tensorflow == 1.13.2
If you have your own datasets, you can set the path in `main.py`.If you only try to run the project, you can contact me to require related datasets by email: 346311816@qq.com.
