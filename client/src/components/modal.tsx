"use client";
import { Modal, ModalContent, ModalHeader, ModalBody, ModalFooter } from "@nextui-org/modal";
import React, { useState, useEffect } from "react";

export default function ModalComponent() {
  const [isOpen, setIsOpen] = useState(false);
  const [img, setImg] = useState("ERROR");
  const [name, setName] = useState("이미지가 전송되지t 않았습니다.");

  const onOpen = () => setIsOpen(true);
  // const onOpen = () => console.log("dd");
  const onClose = () => setIsOpen(false);

  useEffect(() => {
    if (isOpen) { // 모달이 열릴 때만 데이터 가져오기
      fetch("서버_API") // 여기에 API URL 넣기
        .then(response => response.json())
        .then(data => {
          setImg(data.imageUrl); // 서버 응답에서 이미지 URL 설정
          setName(data.name); // 서버 응답에서 이름 설정
        })
        .catch(error => console.error("Error fetching data:", error));
    }
  }, [isOpen]); // isOpen이 변경될 때마다 실행

  const saveImageAsJPG = () => {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    const canvasImg = new Image();
    canvasImg.src = img;
    canvasImg.onload = () => {
      // 캔버스 크기를 이미지 크기에 맞추기
      canvas.width = 400;
      canvas.height = 600;
      
      if (ctx) {
        // 이미지를 캔버스에 그리기
        ctx.drawImage(canvasImg, 0, 0);
        // 이름 텍스트 추가
        ctx.font = "bold 24px Arial"; // 폰트 설정
        ctx.fillStyle = "black"; // 텍스트 색상
        ctx.textAlign = "center"; // 텍스트 정렬
        ctx.fillText(name, canvas.width / 2, canvas.height - 30); // 텍스트 그리기 (중앙 하단)
      } else {
        console.log("canvas Error")
      }

      // JPG 형식으로 데이터 URL 생성
      const link = document.createElement("a");
      link.download = `${name}.jpg`; // 다운로드할 파일 이름 설정
      link.href = canvas.toDataURL("image/jpeg");
      link.click();
    };
  };

  return (
    <>
      <button
      onClick={onOpen} 
      className="px-4 py-2 bg-blue-500 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-opacity-75">
        Open Modal
      </button>

      <Modal isOpen={isOpen} onOpenChange={setIsOpen}>
        df
        <ModalContent className="bg-red-200 shadow-lg rounded-sm overflow-hidden p-6 w-[500px] h-[600px] mx-auto">
          {() => (
            <>
              <ModalHeader className="flex justify-end gap-2 pt-4 border-t">
                <button 
                className="px-2 bg-red-600 text-black font-bold rounded-lg shadow-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-400 focus:ring-opacity-75" 
                onClick={onClose}
                >
                X
              </button>
              </ModalHeader>
              <ModalBody className="text-sm h-[400px] text-gray-700 space-y-4">
                <div className="flex flex-col h-[100vh]">
                  <div className="h-[80%]"><img src={img} alt="ai이미지"/></div>
                  <div className="flex h-[20%] justify-center items-center font-bold text-2xl">{name}</div>
                </div>
              </ModalBody>
              <ModalFooter className="flex flex-row justify-around text-xl font-bold text-gray-900 border-b">
                  <button onClick={saveImageAsJPG}>저장</button>
                  <div>공유</div>
              </ModalFooter>
            </>
          )}
        </ModalContent>
      </Modal>
    </>
  );
}
