import { ImageListComponent } from "@/components/image_list_component";

export default function Home() {

  const itemData = [
    {
      img: "https://images.unsplash.com/photo-1549388604-817d15aa0110",
      title: "Bed",
      text: "이건 침대야"
    },
    {
      img: "https://images.unsplash.com/photo-1525097487452-6278ff080c31",
      title: "Books",
      text: "이건 책이야"
    },
    {
      img: "https://images.unsplash.com/photo-1523413651479-597eb2da0ad6",
      title: "Sink",
      text: "이건 몰라야"
    },
    {
      img: "https://images.unsplash.com/photo-1563298723-dcfebaa392e3",
      title: "Kitchen",
      text: "이건 부엌이야야"
    },
    {
      img: "https://images.unsplash.com/photo-1588436706487-9d55d73a39e3",
      title: "Blinds",
      text: "이건 침대야 침대침야"
    },
    {
      img: "https://images.unsplash.com/photo-1574180045827-681f8a1a9622",
      title: "Chairs",
      text: "이건 허먼밀러야"
    },
    {
      img: "https://images.unsplash.com/photo-1530731141654-5993c3016c77",
      title: "Laptop",
      text: "이건 맥북야"
    },
    {
      img: "https://images.unsplash.com/photo-1481277542470-605612bd2d61",
      title: "Doors",
      text: "이건 문이야야"
    },
    {
      img: "https://images.unsplash.com/photo-1517487881594-2787fef5ebf7",
      title: "Coffee",
      text: "이건 커피일지도야야"
    },
    {
      img: "https://images.unsplash.com/photo-1516455207990-7a41ce80f7ee",
      title: "Storage",
      text: "이건 저장고일지도야"
    },
    {
      img: "https://images.unsplash.com/photo-1597262975002-c5c3b14bbd62",
      title: "Candle",
      text: "이건 침대야ㅈㄷㄹㄷㅈㄹㄷㅈㄹㄷㅈㄹㅈㄷㄹㅈㄹㅈㄷㄹㅈㄹㅈㄷㄹㅈㄷㄹㄷㅈㄹㄷㅈㄹㄷㅈ"
    },
    {
      img: "https://images.unsplash.com/photo-1519710164239-da123dc03ef4",
      title: "Coffee table",
      text: "이건 침대야"
    },
  ];


  return (
    <div className="w-full h-max flex justify-center items-center">
      <ImageListComponent imageDatas={itemData}/>
    </div>
  );
}
