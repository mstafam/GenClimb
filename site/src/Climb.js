function Climb({ frames, circles, layoutInfo }) {
  // Display AI art
    if (!frames) {
      return (
        <img
          src='./assets/climber_bg.jpg'
          alt='Climber Art'
          className="h-auto max-w-full"
          width={500}
          height={700}
        />
      );
    } else {
      // Set variables
      let layoutName = layoutInfo['name'];
      let layoutWidth = layoutInfo['width'];
      let layoutHeight = layoutInfo['height'];
      let layoutEdgeLeft = layoutInfo['edge_left'];
      let layoutEdgeRight = layoutInfo['edge_right'];
      let layoutEdgeBottom = layoutInfo['edge_bottom'];
      let layoutEdgeTop= layoutInfo['edge_top'];
  
      let layoutPath = `./assets/${String(layoutName).replace(/\s+/g, "")}.png`;
  
      let xSpacing = layoutWidth / (layoutEdgeRight - layoutEdgeLeft);
      let ySpacing = layoutHeight / (layoutEdgeTop - layoutEdgeBottom);
      
      // Create each circle element for the holds
      const getCircles = () => {
        return circles.map((el, index) => {
          const [x, y, color] = el;
          let xPixel = (x - layoutEdgeLeft) * xSpacing;
          let yPixel = layoutHeight - (y - layoutEdgeBottom) * ySpacing;
  
          return (
            <circle
              key={index}
              cx={xPixel}
              cy={yPixel}
              strokeWidth="6"
              stroke={color}
              r={xSpacing * 4}
              strokeOpacity={1}
              fillOpacity='0'
            />
          );
        });
      };
  
      // Display board and circles
      return (
        <svg viewBox={`0 0 ${layoutWidth} ${layoutHeight}`}>
          <image href={layoutPath} alt='Climber Art' />
          {getCircles()}
        </svg>
      );
    }
  }

  export default Climb;