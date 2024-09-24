"use client";
import { Button } from "@nextui-org/button";
import {
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
} from "@nextui-org/modal";
import SaveAltRoundedIcon from "@mui/icons-material/SaveAltRounded";
import ShareSharpIcon from '@mui/icons-material/ShareSharp';

interface Data {
  src: string;
  title: string;
  body: string;
}

interface ModalComponentProps {
  isOpen: boolean;
  onOpenChange: () => void;
  data: Data;
}

export default function ModalComponent({
  isOpen,
  onOpenChange,
  data,
}: ModalComponentProps) {

  const saveImage = () => {
    const canvas = document.createElement("canvas");
    const twoD = canvas.getContext("2d");

    const canvasImg = new Image();
    canvasImg.crossOrigin = "Anonymous"; 
    canvasImg.src = data.src;
    canvasImg.onload = () => {
      // 캔버스 크기를 이미지 크기에 맞추기
      canvas.width = 400;
      canvas.height = 600;

      if (twoD) {
        // 이미지를 캔버스에 그리기
        twoD.drawImage(canvasImg, 0, 0, canvas.width, canvas.height);
        // 이름 텍스트 추가
        twoD.font = "bold 24px Arial"; // 폰트 설정
        twoD.fillStyle = "black"; // 텍스트 색상
        twoD.textAlign = "center"; // 텍스트 정렬
        twoD.fillText(data.title, canvas.width / 2, canvas.height - 30); // 텍스트 그리기 (중앙 하단)
      } else {
        console.log("canvas Error");
      }

      // JPG 형식으로 데이터 URL 생성
      const link = document.createElement("a");
      link.download = `${data.title}.jpg`; // 다운로드할 파일 이름 설정
      link.href = canvas.toDataURL("image/jpg");
      link.click();
    };
  };

  const shareImage = () => {

  }
  return (
    <>
      <Modal
        isOpen={isOpen}
        onOpenChange={onOpenChange}
        scrollBehavior={"inside"}
      >
        <ModalContent>
          {() => (
            <>
              <ModalHeader className="flex justify-end gap-2 pt-4"></ModalHeader>
              <ModalBody>
                <img src={data.src} alt="ai이미지" />
                <h1 className="w-full text-center font-extrabold text-5xl">{data.title}</h1>
              </ModalBody>
              <ModalFooter className="flex justify-around">
                <Button color="danger" endContent={<ShareSharpIcon />} onClick={shareImage}>share</Button>
                <Button color="success" endContent={<SaveAltRoundedIcon />} onClick={saveImage}>
                  Save
                </Button>
              </ModalFooter>
            </>
          )}
        </ModalContent>
      </Modal>
    </>
  );
}
