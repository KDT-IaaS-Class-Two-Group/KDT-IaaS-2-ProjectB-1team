import Image from "next/image";
import { useState } from "react";

const ImageUploader = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      setImagePreview(URL.createObjectURL(file));
    }
  };

  const handleSubmit = async () => {
    if (!selectedImage) return;

    setLoading(true);
    const formData = new FormData();
    formData.append("image", selectedImage);

    try {
      const res = await fetch("http://0.0.0.0:8000/process-image", {
        method: "POST",
        body: formData,
      });
      console.log(res);
      if (res.ok) {
        const data = await res.blob();
        const imageUrl = URL.createObjectURL(data);
        setImagePreview(imageUrl);
      } else {
        console.error("실패");
      }
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input type="file" accept="image/*" onChange={handleImageChange} />
      {imagePreview && (
        <Image src={imagePreview} alt="Preview" width={200} height={200} />
      )}
      <button onClick={handleSubmit}>확인</button>
      {loading && <p>로딩 중...</p>}
    </div>
  );
};
export default ImageUploader;
