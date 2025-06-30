import { useRef, useState } from "react";

export default function ImageUploader({ onEvaluate = f => f, onReset = f => f }) {
    const [image, setImage] = useState(null);
    const fileInputRef = useRef();

    const handleImageSelect = (event) => {
        const file = event.target.files[0];
        if (!file) return;
        const imageUrl = URL.createObjectURL(file);
        setImage({ file, url: imageUrl })
    };

    const handleUploadClick = () => fileInputRef.current.click();

    const handleEvaluateClick = () => {
        if (onEvaluate && image) {
            onEvaluate(image.file);
        };
    };

    const handleResetClick = () => {
        setImage(null);
        if (fileInputRef.current) fileInputRef.current.value = "";
        onReset();
    };

    return (
        <div style={styles.wrapper}>
            <div style={styles.previewBox}>
                {image ? (
                    <img src={image.url} alt="preview" style={styles.image} />
                ) : (
                    <span style={styles.placeholderText}>No image selected</span>
                )}
            </div>

            <input 
                type="file"
                accept="image/*"
                onChange={handleImageSelect}
                ref={fileInputRef}
                style={{ display: "none" }}
            />

            <button onClick={image ? handleEvaluateClick : handleUploadClick} style={styles.button}>
                {image ? "Evaluate image" : "Upload image"}
            </button>
            {image && <button onClick={handleResetClick} style={styles.button}>Reset</button>}
        </div>
    );
};

const styles = {
    wrapper: {
        height: "100%",
        margin: "1rem",
        border: "5px solid #007bff",
        borderRadius: "10px",
        padding: "1rem",
        backgroundColor: "#eee",
        textAlign: "center"
    },
    previewBox: {
        width: "500px",
        height: "300px",
        margin: "0 auto 1rem",
        border: "2px dashed #aaa",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        backgroudColor: "#f9f9f9"
    },
    placeholderText: {
        color: "#888",
        fontStyle: "italic"
    },
    image: {
        maxWidth: "100%",
        maxHeight: "100%",
        objectFit: "contain"
    },
    button: {
        margin: "0px 5px",
        padding: "0.6rem 1.2rem",
        fontSize: "1rem",
        cursor: "pointer",
        backgroundColor: "#007bff",
        color: "#fff",
        border: "none",
        borderRadius: "4px"
    }
};