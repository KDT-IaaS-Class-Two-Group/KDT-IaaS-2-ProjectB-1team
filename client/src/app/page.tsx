'use client';

import React, { useState, useRef } from 'react';
import NavBar from '../components/NavBar';
import ImageGallery from '../components/ImageGallery';
import FloatingButton from '../components/FileUploader';
import { ImageListComponent } from '../components/image_list_component'; // 이미지 리스트 컴포넌트 가져오기
import ModalComponent from '@/components/modal';

export default function Home() {
  const [isLoading, setIsLoading] = useState<boolean>(false); // 로딩 상태
  const [images, setImages] = useState<string[]>([]); // 업로드된 이미지들을 저장할 상태
  const fileInputRef = useRef<HTMLInputElement>(null);
  const controllerRef = useRef<AbortController | null>(null); // 업로드 취소용 AbortController
  const uploadTimeoutRef = useRef<NodeJS.Timeout | null>(null); // 업로드 시뮬레이션 취소용 Timeout Ref

  const [isOpen, setIsOpen] = useState<boolean>(true);

  const data = {
    src: "https://images.unsplash.com/photo-1549388604-817d15aa0110",
    title: "도미상",
    body: "테스트"
  }

  // 파일 선택 시 처리하는 함수
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      setIsLoading(true); // 로딩 시작
      const fileArray = Array.from(files);

      // 업로드 요청 취소를 위한 AbortController 생성
      controllerRef.current = new AbortController();

      // 2초 후에 업로드가 완료되는 시뮬레이션
      uploadTimeoutRef.current = setTimeout(() => {
        const newImages = fileArray.map((file) => URL.createObjectURL(file)); // 파일 URL 생성
        setImages((prevImages) => [...prevImages, ...newImages]); // 기존 이미지에 새 이미지 추가
        setIsLoading(false); // 로딩 완료
      }, 2000);
    }
  };

  // 업로드 취소 함수
  const handleCancelUpload = () => {
    if (controllerRef.current) {
      controllerRef.current.abort(); // 업로드 요청 취소
    }

    if (uploadTimeoutRef.current) {
      clearTimeout(uploadTimeoutRef.current); // 시뮬레이션 중인 업로드 취소
    }

    setIsLoading(false); // 로딩 상태 종료
    console.log('업로드가 취소되었습니다.');
  };

  // 파일 선택 창 열기
  const handleIconClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  // 이미지 리스트 데이터
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
    <>
      <NavBar />
      {/* 네비게이션 바와 겹치지 않게 상단에 패딩 추가 */}
      <div className="pt-20">
        {/* 업로드된 이미지 갤러리 */}
        <ImageGallery images={images} />

        {/* 이미지 리스트 컴포넌트 */}
        <div className="w-full h-max flex justify-center items-center mt-8">
          <ImageListComponent imageDatas={itemData} />
        </div>
      </div>

      {/* 오른쪽 아래에 원형 버튼으로 아이콘 표시 */}
      <FloatingButton
        isLoading={isLoading}
        handleCancelUpload={handleCancelUpload}
        handleIconClick={handleIconClick}
      />

      {/* 숨겨진 파일 입력 */}
      <input
        type="file"
        ref={fileInputRef}
        className="hidden"
        onChange={handleFileChange}
      />
      <ModalComponent isOpen={isOpen} onOpenChange={() => {setIsOpen((prev) => !prev)}} data={data}/>
    </>
  );
}

