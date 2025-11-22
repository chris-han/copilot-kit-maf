import { Star as LucideStar, LucideProps } from 'lucide-react';

const ShootingStar = ({ className, ...props }: LucideProps) => {
  return <LucideStar className={className} {...props} />;
};

export default ShootingStar;